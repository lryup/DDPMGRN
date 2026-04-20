from regdiffusion import data
from regdiffusion import train

import numpy as np

benchmark_list=['1000_STRING']

Lsim_tmps=[0.1]

for Lsim in Lsim_tmps:
    for kk in benchmark_list:
        #下面是所有数据集
        big_list={}
        big_list_sum = {}
        big_list_sum2 = {}
        for jj in range(1):
            import pandas as pd
            import time
            benchmark_setting = kk
            temp_list = []
            new_list = []
            temp_list_sum = []
            new_list_sum = []
            temp_list_sum2 = []
            new_list_sum2 = []
            # data_list = ['hHep','mHSC-E']
            data_list = ['hHep']
            # data_list = ['hammond_male_p100']#
            # data_list = ['mHSC-E', 'mHSC-GM']
            for data_tmp in data_list:
                list_result=[]
                list_result_sum = []
                list_result_sum2 = []
                bl_dt, bl_gt = data.load_beeline(
                    benchmark_data=data_tmp, #hESC,hHep,mDC,mESC,mHSC-E,mHSC-GM,mHSC-L
                    benchmark_setting=benchmark_setting#500_STRING,1000_STRING,1000_ChIP-seq,500_ChIP-seq,1000_Non-ChIP,500_Non-ChIP
                )
                print(str(jj)+':'+kk+':'+data_tmp)
                print(data)
                from regdiffusion import evaluator
                start_time = time.time()

                rd_trainer = train.DDPMGRN(bl_dt.X, n_steps=1000,end_noise=0.05)
                # rd_trainer.train()
                out_z = rd_trainer.train(Lsim=Lsim)  # 设置loss_sim的参数

                end_time = time.time()
                total_time = end_time - start_time  # seconds

                evaluator = evaluator.GRNEvaluator(bl_gt, bl_dt.var_names)
                inferred_adj = rd_trainer.get_adj()
                # evaluator.evaluate(inferred_adj)
                ppi_auc = evaluator.evaluate(inferred_adj)
                print(ppi_auc)

                ppi_auc = [ppi_auc['AUROC'],ppi_auc['AUPR'],ppi_auc['AUPRR'], ppi_auc['EP'],ppi_auc['EPR']]


                # print(ppi_auc)
                list_result.append(kk)
                list_result.append(data_tmp)
                list_result.append(total_time)
                list_result.extend(ppi_auc)  #
                new_list.append(list_result)


            temp_list.extend(new_list)
            big_list[jj] = temp_list
            df = pd.DataFrame()
            for key, value in big_list.items():
                temp_df = pd.DataFrame(value)
                df = pd.concat([df, temp_df], axis=1)
        df.to_csv('DDPMGRN.csv', mode='a', index=False, header=False)
