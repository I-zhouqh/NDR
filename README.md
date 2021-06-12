# NDR

## simulation
配置在config.py里面修改
然后python运行 main.py，会产生data文件夹，并在data文件夹下存储很多xlsx文件。
other_sdr.R会对保存下来的数据做其他SDR方法（SIR,SAVE,MAVE等），并储存xlsx文件。（此步可以跳过）
summary_result.py，会对所有excel进行汇总（可选择是否汇总other SDR方法）。

## realdata
python realdata.py
需要在realdata.py文件里面定制好数据的读取方式
实际上，训练已经被比较好地封装为了
sdr_model.train(x_train,y_train)
sdr_model.test(x_test,y_test)
的方式
