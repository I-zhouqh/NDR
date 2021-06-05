# install.packages("tidyverse")
# install.packages("dr")
# install.packages("readr")
# install.packages("readr")
# install.packages("openxlsx")
# install.packages("xlsx")
# install.packages("MAVE") 


install.packages("rJava")
options(java.parameters = "-Xmx4096m")


library(dr)
library(readr)
library(tidyverse)
library(openxlsx)
library(MAVE)

path = "D:\\study\\SDR\\work_combine\\finalSimulation_d4\\final_d4\\n500"
save_path = paste(path, "_res\\", sep="")


Other_Sdr_Model = function(filepath, save_path)
{
  setwd(filepath)
  # 如果不存在保存文件的文件夹，创建
  if(dir.exists(save_path))
  {
    print("Begin work...")
  }
  else
  {
    print("Create filepath and begin work...")
    dir.create(save_path)
  }
  
  
  num_file = length(dir())
  for(i in 1:num_file)
  {
    filename = c(dir())[i]
    if(length(grep("xlsx", filename) > 0))
    {
      print(paste("input file no",i))
      print(filename)
      ##################################
      # 执行读取数据等一系列操作
      {
        # 读入数据，需要把最后一列变成y
        data = readWorkbook(filename, sheet = 'Data', colNames = FALSE, skipEmptyRows = FALSE,  startRow = 1)
        if(nrow(data) != 500){
          data = readWorkbook(filename, sheet = 'Data', colNames = TRUE, skipEmptyRows = FALSE,  startRow = 1)
        }
        
        colnames(data)[length(colnames(data))]  = 'y'
        # 这里的truespace其实有很多不必要的维度，后续还需要根据subspace来调整维度
        true_space = readWorkbook(filename, sheet = 'true_space', colNames = TRUE, skipEmptyRows = FALSE,  startRow = 1)
        true_space = do.call(cbind, true_space)
        x_names = colnames(data)[1:ncol(data)-1]
        y = data[,'y']
        x = data[, x_names]
        x = do.call(cbind, x)
        print(nrow(data))
        print(ncol(data))
        
        # 设定参数，找个df保存结果
        alpha = 0.05
        methods = c('sir','save','phdy','phdres','ire','meanOPG','meanMAVE','csMAVE','csOPG','KSIR')
        res_sum_df = data.frame(method = NULL,
                                best_k = NULL,
                                beta_estimate_str = NULL,
                                space_estimate_str = NULL,
                                distance = NULL,
                                scale_of_matrix = NULL)
        
        # 对每种方法计算结果, 其中ire的结构和其他不一样
        for (method in methods)
        {
          if(method == 'ire')
          {
            obj = dr(y ~ x, numdir = ncol(x), method = "ire")
            res = summary(obj)
            
            # 找到最优的k
            for (i in c(1:nrow(res$test[3])))
            {
              best_k = 0
              if(res$test[i+1, 3] > alpha)
              {
                best_k = i
                break
              }
            }
            
            # 子空间结果, 前面的索引是最优k
            tmp = res$result[best_k][[1]]$B
            sub_space = tmp %*% solve(t(tmp)%*%tmp) %*% t(tmp)
          }
          else if(method %in% c('meanOPG','meanMAVE','csMAVE','csOPG','KSIR'))
          {
            obj <- mave(y ~ x, method = method)
            dim_res = mave.dim(obj, max.dim = ncol(x))
            best_k = which.min(dim_res$cv)
            tmp = coef(obj, best_k)
            sub_space = tmp %*% solve(t(tmp)%*%tmp) %*% t(tmp)
          }
          # else if(method == "mave_meanopg")
          # {
          #   obj <- mave(y ~ x, method = 'meanopg')
          #   dim_res = mave.dim(obj, max.dim = ncol(x))
          #   best_k = which.min(dim_res$cv)
          #   tmp = coef(obj, best_k)
          #   sub_space = tmp %*% solve(t(tmp)%*%tmp) %*% t(tmp)
          # }
          # else if(method == "mave_ksir")
          # {
          #   obj <- mave(y ~ x, method = 'ksir')
          #   dim_res = mave.dim(obj, max.dim = ncol(x))
          #   best_k = which.min(dim_res$cv)
          #   tmp = coef(obj, best_k)
          #   sub_space = tmp %*% solve(t(tmp)%*%tmp) %*% t(tmp)
          # }
          else
          {
            obj = dr(y ~ x, numdir = ncol(x), method = method)
            res = summary(obj)
            
            # 找到最优的k, 然后重新训练降维
            for (i in c(1:nrow(res$test[3])))
            {
              best_k = 0
              if(res$test[i+1, 3] > alpha)
              {
                best_k = i
                break
              }
            }
            obj = dr(y ~ x, numdir = best_k, method = method)
            res = summary(obj)
            # 找到最优降维子空间
            tmp = res$evectors
            sub_space = tmp %*% solve(t(tmp)%*%tmp) %*% t(tmp)
          }
          
          # 最后调整truespace的维度，然后一起算distance
          true_space = true_space[1:nrow(sub_space), 1:ncol(sub_space)]
          # 特征值出现复数
          distance = tryCatch(max(eigen(true_space - sub_space)$values),
                              error = function(dis){return(1)},
                              finally = function(dis){return(max(eigen(true_space - sub_space)$values))})
          # 先保存此数据下的best_k和distance
          res_sum_df[method, 'best_k'] = best_k
          res_sum_df[method, 'distance'] = distance
          
          # 保存beta和sub_space为字符串
          beta = tmp
          beta_estimate_str = ""
          space_estimate_str = ""
          # 先保存beta， beta要判断是不是1维的
          tmp_str = ""
          if (is.null(nrow(beta)))
          {
            for (i in c(1:length(beta)))
            {
              tmp_str = paste(tmp_str, beta[i])
            }
            beta_estimate_str = tmp_str
          }
          else
          {
            for (i in c(1:nrow(beta)))
            {
              
              for (j in c(1:ncol(beta)))
              {
                tmp_str = paste(tmp_str, as.character(beta[i, j]), sep = " ")
              }
              tmp_str = paste(tmp_str, "\n")
            }
            beta_estimate_str = tmp_str
          }
          
          
          tmp_str = ""
          for (i in c(1:nrow(sub_space)))
          {
            for (j in c(1:ncol(sub_space)))
            {
              tmp_str = paste(tmp_str, as.character(sub_space[i, j]), sep = " ")
            }
            tmp_str = paste(tmp_str, "\n")
          }
          space_estimate_str = tmp_str           
          
          res_sum_df[method, 'beta_estimate_str'] = beta_estimate_str
          res_sum_df[method, 'space_estimate_str'] = space_estimate_str
          res_sum_df[method, 'method'] = method
          res_sum_df[method, 'scale_of_matrix'] = sum(sub_space**2)**0.5
          
          # 保存一下NDR结果
          
        }    
      }
      
      # 保存结果
      list_of_datasets <- list("OtherSdrRes" = res_sum_df)
      write.xlsx(list_of_datasets, file = paste(save_path, "OtherSdr", filename))
    }
    else
    {
      print("This file is not EXCEL file!") 
    }
  }
}

Other_Sdr_Model(path, save_path)















# 
# 
# setwd(path)
# data = 1
# data = readWorkbook("Exp0519_0102__.xlsx", sheet = 'DR_metrics_all_Method', colNames = FALSE)
# data

# true_space = readWorkbook("Exp00_02__.xlsx", sheet = 'true_space', colNames = TRUE, skipEmptyRows = FALSE,  startRow = 1)
# true_space = do.call(cbind, true_space)

# true_space[1:3, 1:3]
# 
# 
# nrow(data)
# ncol(data)
# 
# colnames(data)[length(colnames(data))]  = 'y'





