"""
Further concern:

gamma的最优化是随机投点的

用每一层（比如w1）的频率分布图来visualize参数的变化。挺好。画均值标准差，也不错，求和感觉有点太笨了。or plot unit值也可以 (tensorboard ?)
gamma的图何不如画在一张图里。

don't use sigmoid, try relu ?

输入应该以0为中心。也就是应该标准化。!!!!!!!!!!!!

均匀分布与正态分布

神经网络参数只更新3次就结束了，会不会有点问题。

调整BN、dropout、lr？

没有分batch

hidden state的数量

lr ?

如果r是随机投点的话，设置初值没什么用！
如果r是梯度下降，那设置初值才有一些用！

reduction=sum



0313代办
1.检查self.eye的设置是不是全部合理  ok
2.检查数值稳定性
3.还是写一个日志吧。logging一下。不然好像太笨了
4.第二次剥离降维时，网络还是要更新吗。
5.何种论文来参照一下，科学论文的方式。
6.这个I-rr'到底是啥。
8.结构到底是什么。  做到什么地步-----

1.introduction literature review
2.介绍我的方法
3.simulation加结果  8-10页



#随机投点时，那些个点的分布。是正弦曲线的样子

#固定数据，固定r的生成时，居然还是出现了不一样的收敛结果，并且收敛结果大大相反！（虽然都是正确的，第一次先降r5，第二次先降r4)
问题出在神经网络参数的随机初始化


# 如果一开始tensor就建立在cuda上？

如果x1,x2,x3都有用，但我们强行投影，能否得出，强行投影的loss大于直接网络的loss？这个很重要。

#停止降维-并不足以识别出他


# 信噪比真的很影响，标准差之比10:2的时候稳的一匹，10:5的时候还将就，10:10彻底拉胯。
# 学习率也真的很重要，在某次结果不好是因为学习率太小，晕了


# 不正交的问题！
就比如降10次维吧，降到第五次的时候，r5和r1,r2,r3,r4的正交性保持的很好，但是到后面，就没那么好了，r8和r2的内积可以达到0.2了（一般是1e-5这个量级）
算法没有问题的，关键就在于比如 r2=(I-r1r1')r0，这么一算，理想情况下r1'r2是等于0，但是不是严格等于呀！慢慢地误差就积累了。
以往那个误差我们可以强制要求r的模长为1，现在这个误差咋办？总不能强制去要求eye的模长吧。可以吗。

check 正交性画个矩阵

# model给出的yhat是batch_size*1的结果
# 你的y也必须要是batch_size*1，而不能是batch_size!!!不然的话，去算yhat与y的损失函数时，会张成一个矩阵去算，却不会报错，太坑了！

#W的容忍度目前还是太大了吧，不要平滑了，比最小大几次就爬，另外可不可以返回原来的状态？
#对，不要平滑。比之前的最小点高三次就爬（连续高三次还是累计高三次？感觉连续高三次好一点，累计高三次有点小离谱，万一某一次是偶然跌到谷底，那不就凉了。



"""

import torch
import copy

import datetime
from torch import nn
import numpy as np
import torch.nn.functional as F
from service.utils import MLP





class Monitor:
    def __init__(self,r_patience=100,W_patience=3,W_MAperiod=5):
        """

        :param r_patience: gamma的模连续稳定多少次，就不等了，停止更新gamma
        :param W_patience: eval_loss连续上升多少次，就不等了，停止更新W。如果W_patience==-1，则不对W做限制
        """

        self.r_patience=r_patience
        self.W_patience=W_patience
        self.W_MAperiod=W_MAperiod


        self.gamma_only = []  #所有gamma的统计信息
        self.update_r_times = 1    #服务于gamma_only的变量，更新了几次r
        self.all_stats=[]     #所有统计信息

        self.num_of_dimension_reduction=1  #目前是第几次降维，每train一次就代表一次降维，这个东西就会加一。
        self.all_gamma_and_eye={}  #所有gamma和eye的统计信息  {1: {'r':tensor, 'eye':tesor} }

    def check_orthogonal(self):
        """
        检查每个gamma之间的正交性，检查用，不参与运算
        :return: p*p矩阵，第[i,j]个元素代表ri和rj的内积
        """
        gammas=[]
        for k in sorted(self.all_gamma_and_eye):
            gammas.append(self.all_gamma_and_eye[k]['gamma'])
        gamma_matrix=torch.cat(gammas,dim=1)
        return gamma_matrix.t().matmul(gamma_matrix)

    def reset_monitor_variable(self):
        self.r_repeat_times=0  #r的模当前维持了多少次，与r_patience比较
        self.update_r_times=1

        self.W_up_times=0   #eval_loss当前上升了多少次，与W_patience比较
        self.evalloss_MovingAverage=[]  #滑动平均所用的list
        self.evalloss_nowMin=np.Inf  #目前的最小值

    def check_update_r_state(self,cos_distance):
        """
        停止更新gamma的逻辑：当连续r_patience次与上一次的cos距离非常近时，就停止更新gamma！注意是连续的
        :param cos_distance: 这一次r与上一次r的cos距离
        """

        r_state=False
        if self.r_repeat_times <= self.r_patience:  # 如果连续r_patience个回合不动弹，那就停止更新gamma没有机会继续更新gamma
            r_state=True
            if np.abs(np.abs(cos_distance) - 1) < 1e-5:
                self.r_repeat_times += 1
            else:
                self.r_repeat_times =0
        return r_state



    def check_update_W_state2(self, eval_loss):
        """
        停止更新W的逻辑：比之前的最小值连续用eval_loss来判断W的停止。如果没有eval_loss就无法停止W的更新，只能更新到最大次数。
        当eval_loss连续W_patience次大于当前最小值时，则停止更新W。当然为了防止随机波动的影响，这个eval_loss是十次滑动平均的eval_loss。
        :param eval_loss:
        :param W_MAperiod:
        :return:
        """

        if eval_loss is None or self.W_patience == -1:
            # 没有eval_loss，那W没有停止标准，W只能继续更新
            # 如果W_patience==-1，意味着W没有停止标准
            return True
        if self.W_up_times > self.W_patience:  # 如果已经超过限制，就永远不得更新
            return False

        if eval_loss < self.evalloss_nowMin:
            self.evalloss_nowMin = eval_loss
            self.W_up_times = 0
        else:
            self.W_up_times += 1

        return True

    def check_update_W_state(self, eval_loss):
        """
        停止更新W的逻辑：用eval_loss来判断W的停止。如果没有eval_loss就无法停止W的更新，只能更新到最大次数。
        当eval_loss连续W_patience次大于当前最小值时，则停止更新W。当然为了防止随机波动的影响，这个eval_loss是十次滑动平均的eval_loss。
        :param eval_loss:
        :param W_MAperiod:
        :return:
        """

        if eval_loss is None or self.W_patience == -1:
            # 没有eval_loss，那W没有停止标准，W只能继续更新
            # 如果W_patience==-1，意味着W没有停止标准
            return True
        if self.W_up_times >= self.W_patience:  # 如果已经超过限制，就永远不得更新
            return False
        if len(self.evalloss_MovingAverage) < self.W_MAperiod:  # 滑动平均阶数还没到，那还是要继续更新
            self.evalloss_MovingAverage.append(eval_loss)
            return True

        self.evalloss_MovingAverage.append(eval_loss)
        self.evalloss_MovingAverage.pop(0)

        now_eval_loss = np.mean(self.evalloss_MovingAverage)
        if now_eval_loss < self.evalloss_nowMin:
            self.evalloss_nowMin = now_eval_loss
            self.W_up_times = 0
        else:
            self.W_up_times += 1

        return True


    def log_all_gamma_and_eye(self,r,eye,NNW_weight):

        self.all_gamma_and_eye[self.num_of_dimension_reduction] =       {'gamma':r,
                                                                         'eye':eye,
                                                                         'train_loss':self.all_stats[-1]['train_loss'],
                                                                         'eval_loss': self.all_stats[-1]['eval_loss'],
                                                                         'NNW_weight':NNW_weight.copy()
                                                                         }
        self.num_of_dimension_reduction+=1

    def log_gamma_only(self,r,true_p):
        gamma_only_stat = {}
        gamma_only_stat['epoch'] = self.update_r_times
        gamma_only_stat['num_of_dimension_reduction'] = self.num_of_dimension_reduction  # 第几次剥离
        for i in range(true_p):  # 储存r
            name = "r" + str(i + 1)
            gamma_only_stat[name] = r[i].item()
        for i in range(true_p):  # 储存r
            name = "r" + str(i + 1)+"平方"
            gamma_only_stat[name] = (r[i].item())**2

        self.gamma_only.append(gamma_only_stat)


    def log_all_stats(self,i, sdr_model, cos_distance,r_state,W_state, train_loss, eval_loss=None):

         # 将一个epoch的信息写进字典
        p = sdr_model.r.shape[0]
        stat = {}
        stat['epoch'] = i
        stat['num_of_dimension_reduction'] = self.num_of_dimension_reduction  #第几次剥离
        stat['update_r']=r_state
        stat['update_W']=W_state

        for name, parameter in sdr_model.NNWmodel.named_parameters():
            stat[name + '.mean'] = parameter.data.mean().item()
            stat[name + '.std'] = parameter.data.std().item()

        for i in range(p):  # 储存r
            name = "r" + str(i + 1)
            stat[name] = sdr_model.r[i].item()

        stat['cosdistance_versus_last_gamma']=cos_distance

        r_norm=0    #储存r的平方
        for i in range(p):  # 读取r
            name = "r" + str(i + 1)
            stat[name+"平方"] = (sdr_model.r[i].item())**2
            r_norm+=(sdr_model.r[i].item())**2
        stat['r_norm']=r_norm

        stat['train_loss'] = train_loss
        stat['eval_loss'] = eval_loss

        self.all_stats.append(stat)





class SDRmodel:

    def __init__(self,p, NNW_params, DEVICE,seed=4396):

        #self.r = None

        self.seed=seed
        self.true_p = p
        self.DEVICE=DEVICE
        self.NNWmodel = self.get_NNWmodel(NNW_params)

        self.loss_function = NNW_params['loss_function'].to(self.DEVICE)
        # self.optimizer=optimizer
        self.optimizer = torch.optim.Adam(self.NNWmodel.parameters(), lr=NNW_params['learning_rate'],weight_decay=NNW_params['weight_decay'])
        self.cos=nn.CosineSimilarity(dim=0, eps=1e-6)

        self.monitor=Monitor()

    def get_NNWmodel(self,NNW_params):

        torch.manual_seed(self.seed)  # 神经网络的权重初始化也是随机的，这个也挺坑的。
        hidden=NNW_params['hidden']
        if not isinstance(hidden,list):
            hidden=[hidden]
        NNWmodel = MLP([self.true_p]+hidden+[1],activation=NNW_params['activation'])
        NNWmodel.to(self.DEVICE)
        param_names = [name for name, _ in NNWmodel.named_parameters()]
        print(param_names)

        return NNWmodel


    @property
    def get_all_gamma_and_eye(self):
        tmp=self.monitor.all_gamma_and_eye.copy()
        for item in tmp.values():
            item.pop('NNW_weight')
        return tmp   #其实就是返回monitor.all_gamma_and_eye，但是把NNW_weight这个键去掉，这个太长了

    @property
    def get_all_stats(self):
        return self.monitor.all_stats.copy()
    @property
    def get_gamma_only(self):
        return self.monitor.gamma_only.copy()
    @property
    def get_best_DR_space(self):
        stats = self.monitor.all_gamma_and_eye.copy()
        if len(stats)==0:
            raise ValueError
        minLoss=np.Inf
        bestDimensionReduction=0
        #找出最优的降维次数
        for k,v in stats.items():
           if v['eval_loss']<minLoss:
                minLoss=v['eval_loss']
                bestDimensionReduction=k

        return {'method':'SDRNNW',
                'DimensionFound':self.true_p-bestDimensionReduction, #表示找到了几个低秩结构，所以应该是全部维度减去降维次数
                'EstimatedBasis': None,
                'EstimatedSpace':stats[bestDimensionReduction]['eye'].clone().numpy(),
                }




    def check_orthogonal(self):
        return self.monitor.check_orthogonal()

    def set_r(self,r):
        # 设置r，处理完一个维度之后要处理下一个维度，所以r需要被反复设置。
        self.r=r.clone()

    def update_eye(self,r=None):
        """
        剥离多次的情况，见算法。
        为了保证后面的降维维度和之前的降维维度正交。需将前面的方向减掉。等价于替换掉eye。
        如果不指定，就设置为单位eye（这是最刚开始的情况）
        :param r: 已经找好的方向。需要去掉
        :return:
        """
        if r is None:
            self.eye = torch.eye(self.true_p, self.true_p, dtype=torch.float, requires_grad=False).to(self.DEVICE)
        else:
            self.eye=self.eye-r.matmul(r.t())


    def test(self,x, y=None ,state="train"):
        """

        :param x:
        :param y:
        :param state:state="train", "finish"
        train意味着依然处于训练中，此时test就利用当前gamma值来算
        finish意味着已经结束训练，此时test应该根据最优降维次数下的参数计算，降维次数的判断通过self.monitor.all_gamma_and_eye中的eval_loss，该降维次数下的参数也存在这个dict里面
        remind that 调用state=='finish'会将
        :return:
        """
        if state=="train":
            _,loss=self.forward(x,y)
            return loss
        elif state=="finish":
            stats=self.monitor.all_gamma_and_eye.copy()

            minLoss=np.Inf
            bestDimensionReduction=0
            #找出最优的降维次数
            for k,v in stats.items():
                if v['eval_loss']<minLoss:
                    minLoss=v['eval_loss']
                    bestDimensionReduction=k

            tmpModel=copy.deepcopy(self.NNWmodel) #必须是深拷贝！
            tmpModel.load_state_dict(stats[bestDimensionReduction]['NNW_weight'])

            x_star = x.matmul(stats[bestDimensionReduction]['eye'])  # x_star
            yhat = tmpModel(x_star)

            result = (bestDimensionReduction,yhat,)
            if y is not None:
                if yhat.shape != y.shape:
                    raise ValueError("dimension not match !!")
                loss = self.loss_function(yhat, y)
                result += (loss,)
            return result

        else:
            raise ValueError("wrong input for state")


    def forward(self, x, y=None,r=None):
        if r is None:
            r=self.r.clone() #在少部分时候，使用的是自己的r，并不是类的r。当然如果不指明，就认为是用类的r

        x_star = x.matmul(self.eye - r.matmul(r.t()))  # x_star
        yhat = self.NNWmodel(x_star)
        result = (yhat,)
        if y is not None:
            if yhat.shape!=y.shape:
                raise ValueError("dimension not match !!")
            loss = self.loss_function(yhat, y)
            result += (loss,)
        return result

    def update_W(self, x, y, verbose=True):

        steps = 1
        while steps<=3:
            _, loss = self.forward(x, y)  # 第一个返回值是yhat，没用，只需要loss即可。
            if verbose:
                print("update W, loss:", loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # optimizer只会更新网络中的参数，不会更新r

            #new_param = self.NNWmodel.fc1._parameters['weight'].clone()
            #print("w1 norm:",(old_param-new_param).norm().item())
            # print("diff%f"%(new_norm-old_norm) )
            #             if np.abs((new_norm-old_norm))<threshold:
            #                 break
            # old_param = new_param.clone()
            steps += 1

    def cal_l_round_x(self,x,y,r=None):
        #对最原始的那个x的导数

        if r is None:
            r=self.r.clone()   #在少部分时候，使用的是自己的r，并不是类的r。当然如果不指明，就认为是用类的r
        n=x.shape[0]

        ##前向传播
        #eye = torch.eye(self.true_p, self.true_p, dtype=torch.float)
        x_star = x.matmul(self.eye - r.matmul(r.t()))
        x_star.requires_grad = True
        s2 = self.NNWmodel(x_star)

        if s2.shape!=y.shape:
            raise  ValueError ("dimension not match!!")

        loss = self.loss_function(s2, y)

        ##反向传播计算梯度，但是并不更新，目的只是去得到x_star的梯度。
        self.optimizer.zero_grad()
        loss.backward()

        l_round_x_star = x_star.grad  # n*p
        l_round_x=l_round_x_star.matmul( self.eye - r.matmul(r.t()) ) # n*p
        return l_round_x

    def cal_l_round_r(self, x, y,r=None):
        if r is None:
            r=self.r.clone()   #在少部分时候，使用的是自己的r，并不是类的r。当然如果不指明，就认为是用类的r
        n=x.shape[0]

        ##前向传播
        #eye = torch.eye(self.true_p, self.true_p, dtype=torch.float)
        x_star = x.matmul(self.eye - r.matmul(r.t()))
        x_star.requires_grad = True
        s2 = self.NNWmodel(x_star)

        if s2.shape!=y.shape:
            raise ValueError("dimension not match!!")

        loss = self.loss_function(s2, y)

        ##反向传播计算梯度，但是并不更新，目的只是去得到x_star的梯度。
        self.optimizer.zero_grad()
        loss.backward()

        l_round_x_star = x_star.grad

        ## 有了x_star的梯度，即去计算得出r的梯度
        l_round_r = torch.zeros(self.true_p, 1, dtype=torch.float).to(self.DEVICE)
        eye = torch.eye(self.true_p, self.true_p, dtype=torch.float).to(self.DEVICE)   #这里用的eye不是self.eye，看看梯度就知道了！
        for i in range(n):
            x_star_i_round_r = -( r.matmul(x[i, :].view(1, -1)) + r.t().matmul(x[i, :]) * eye  )  #负号为什么能忘
            l_i_round_r = x_star_i_round_r.matmul(l_round_x_star[i, :].reshape(self.true_p, 1))
            l_round_r += l_i_round_r
        return l_round_r

    def update_r(self, x, y, verbose=True, threshold=1e-5):
        # 更新参数r，此时W是不动的，相当于

        def find_best_t(h, r,method='random_search',search_points=1000,verbose=True):
            ## 用格点搜索的办法，找到一个最优的t
            ## 也可以梯度下降，但是格点法会简便一下。我认为效果不会差太多
            if method=='random_search':
                search_min = 0
                search_max = 2 * np.pi / h.norm()
                delta = (search_max - search_min) / search_points
                min_value = 1000000
                optimal_t = 0

                for i in range(search_points):
                    t = search_min + i * delta
                    this_r = cal_r_by_t(h, t, r)

                    yhat, this_loss = self.forward(x, y, this_r)
                    this_loss=this_loss.item()
                    if this_loss < min_value:
                        min_value = this_loss
                        optimal_t = t
                if verbose:
                    print("update r, loss:", min_value)
                return optimal_t
            elif method=="GD":
                t=0
                for _ in range(100):
                    f_t=cal_r_by_t(h,t,r)
                    yhat , loss=self.forward(x,y,f_t)
                    loss=loss.item()
                    print(f"in GD for t, round: {_},loss:{loss}")

                    l_round_r=self.cal_l_round_r(x, y, f_t)  # p*1
                    f_t_round_t=  - h.norm()* r * torch.sin(h.norm()*t) +  h*torch.cos(h.norm()*t )  #p*1
                    l_round_t = l_round_r.t().matmul(f_t_round_t)  #标量，t的梯度
                    t -= l_round_t*0.01
                if verbose:
                    print("update r, loss:", loss)
                return t

        def cal_r_by_t(h, t, r):
            result = r * torch.cos(h.norm() * t) + h / h.norm() * torch.sin(h.norm() * t)

            #本来计算出result就ok了，但是因为数值运算的误差，导致result模不严格等于1，与之前的gamma正交性也不能完全保证。
            #尽管这个误差非常小，但是在长期积累之后就相当客观，不可小觑，因此必须每次运算后都严格规范。
            #即规范本次gamma和之前gamma的正交性，规范其模长严格为1.
            tmp=self.eye.matmul(result)
            tmp=tmp/tmp.norm()
            return tmp


        r=self.r.clone()

        if verbose:
            print("start updating r in this round:", r)

        l_round_r = self.cal_l_round_r(x, y, r)
        g = (self.eye - r.matmul(r.t())).matmul(l_round_r)
        h = -g

        k = 0  # 计数项
        while k < 20:  # r还是不要弄太多次吧，时间太久了。虽然一般也不会超过20

            # 找到最优的t
            t = find_best_t(h,r,verbose=verbose)
            # 更新r，同时保留旧值
            old_r = r.clone()
            #print("check orthogonial of h and r",h.t().matmul(r).item())
            r = cal_r_by_t(h, t ,r)


            #记录只有gamma的统计情况
            self.monitor.log_gamma_only(r,self.true_p)


            # 判断终止条件
            # print("r_diff:",(self.r-old_r).norm())
            if (r - old_r).norm() <= threshold:
                break

            g_plus_1 = (self.eye - r.matmul(r.t())).matmul(self.cal_l_round_r(x, y, r))
            h_tao = -old_r * torch.sin(h.norm() * t)*h.norm()  + h  * torch.cos(h.norm() * t)
            g_tao = g - old_r * torch.sin(h.norm() * t) + h / h.norm() * (1 - torch.cos(h.norm() * t)) * h.t().matmul(
                g) / h.norm()
            delta = (g_plus_1 - g_tao).t().matmul(g_plus_1) / g.t().matmul(g)

            # 更新
            h = -g_plus_1 if (k + 1) % (self.true_p - 1) == 0 else (-g_plus_1 + delta * h_tao)
            g = g_plus_1
            k += 1

        if verbose:
            print("final updated r in this round:", r)
            print("cos distance between r and last r:",self.cos(self.r,r).item()  )
        self.r=r.clone()
        self.monitor.update_r_times+=1


    def train_one_dimension(self,x,y,max_exp_times,x_eval=None,y_eval=None, verbose=1):
        """

        :param x:
        :param y:
        :param exp_times: 最大训练次数，gamma的训练次数小于等于此
        :param x_eval:
        :param y_eval:
        :param verbose: 0:不打印任何，1:隔100个epoch打印, 2：全部打印
        :return:
        """
        print(f"start at {str(datetime.datetime.now())}")

        last_r=self.r.clone()  #用于计算cos距离的

        self.monitor.reset_monitor_variable()


        for i in range(max_exp_times ):

            this_epoch_print= verbose==2 or (verbose==1 and i%100==0)

            if this_epoch_print:
                print(f"*********{i}/{max_exp_times}*********")

            cos_distance=self.cos(last_r,self.r).item()  #和上一组gamma之间的cos距离
            last_r=self.r.clone()
            _, train_loss = self.forward(x, y)
            train_loss=train_loss.item()
            eval_loss=None
            if x_eval is not None and y_eval is not None:
                eval_loss = self.test(x_eval, y_eval)
                eval_loss=eval_loss.item()

            r_state = self.monitor.check_update_r_state(cos_distance)
            W_state = self.monitor.check_update_W_state(eval_loss)

            self.monitor.log_all_stats( i, self, cos_distance, r_state, W_state, train_loss, eval_loss )

            if not r_state and not W_state:
                break

            if W_state:
                self.update_W(x, y,verbose=this_epoch_print)
            if r_state:
                self.update_r(x, y,verbose=this_epoch_print)

    def generate_random_r(self,distribution,params,seed=1):
        if distribution == 'normal':
            np.random.seed(seed)
            r = np.random.normal(params['mean'], params['sigma'], (self.true_p,1))
        elif distribution == 'uniform':
            np.random.seed(seed)
            r = np.random.uniform(params['lbound'], params['ubound'], (self.true_p, 1))
        else:
            raise ValueError("distribution %s not avaiable" % distribution)

        r = torch.tensor(r, requires_grad=False, dtype=torch.float).to(self.DEVICE)
        return r

    def train(self,x,y,x_eval,y_eval,max_exp_times=3000,DR_times=None,verbose=1):
        # 如果没有指明降维次数，那就拉满，降维p-1次
        if DR_times is None:
            DR_times=self.true_p-1

        # 初始化第一次降维的r；之后几次降维中，r的初始化由SwitchToNextStrip完成

        r=self.generate_random_r('uniform',{'lbound':-1,'ubound':1},seed=self.seed)
        r = r / r.norm()
        self.set_r(r)
        self.update_eye()

        # 训练
        for DR_iter in range(1, DR_times + 1):  # 降几次维
            print(f"##现在是第{DR_iter}/{DR_times}次降维##")
            self.train_one_dimension(x, y, max_exp_times, x_eval=x_eval, y_eval=y_eval, verbose=verbose)
            self.SwitchToNextStrip(seed_for_r=self.seed + DR_iter)


    def SwitchToNextStrip(self,seed_for_r=None):
        #更新eye
        print("#########SwitchToNextStrip###########")
        r_completed = self.r.clone()
        r_completed = r_completed/r_completed.norm()
        print(r_completed)

        print(self.eye)
        self.update_eye(r_completed)
        print(self.eye)

        #储存gamma和eye的信息
        self.monitor.log_all_gamma_and_eye(r_completed.clone(),self.eye.clone(),self.NNWmodel.state_dict())


        #建立下一个r
        if seed_for_r is None:
            seed_for_r=np.random.randint(0,100000,1).item()
        r = self.generate_random_r('uniform', {'lbound': -1, 'ubound': 1}, seed=seed_for_r)
        ##为了保持正交
        r = self.eye.matmul(r)
        r = r / r.norm()

        for key,value in self.monitor.all_gamma_and_eye.items():
            print(f"compared with gamma {key}, the dot product is {value['gamma'].t().matmul(r).item()} "  )

        self.set_r(r)
        print("#########SwitchEnd###########")




def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

