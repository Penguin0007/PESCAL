import json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, os.path
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
p_ratios=['']

def read(iters,ratio,t,Nodefaultcat=False):
    if not Nodefaultcat and t:
        t=['']+t
    else: pass
    category=["pescal_real_"]
    for typ in t:
        if typ:
            pdf=PdfPages("pescal_real_"+typ+'_'+str(ratio)+".pdf")
        else:
            pdf=PdfPages("pescal_real_"+str(ratio)+".pdf")
        results={}
        for cat in category:
            for i in range(iters):
                i+=1
                if typ:
                    if os.path.exists("./data/"+cat+typ+'_'+str(ratio)+"_"+str(i)+'.json'):
                        with open("./data/"+cat+typ+'_'+str(ratio)+"_"+str(i)+'.json', 'r') as f:
                            results[cat+str(i)]= json.load(f)
                    else:
                        results[cat+str(i)]= results[cat+str(i-1)]
                else:
                    if os.path.exists("./data/"+cat+str(ratio)+"_"+str(i)+'.json'):
                        with open("./data/"+cat+str(ratio)+"_"+str(i)+'.json', 'r') as f:
                            results[cat+str(i)]= json.load(f)
                    else:
                        results[cat+str(i)]= results[cat+str(i-1)]

        for cat in category:
            fqi_rewards, fqi_rewards_running, fqi_loss=[], [], []
            cal_rewards, cal_rewards_running, cal_loss=[], [], []
            cql_rewards, cql_rewards_running, cql_loss=[], [], []
            pescal_rewards, pescal_rewards_running=defaultdict(list),defaultdict(list)

            for i in range(iters):
                i+=1
                fqi_rewards.append(results[cat+str(i)]["fqi_rewards"])
                fqi_rewards_running.append(results[cat+str(i)]["fqi_rewards_cumulate"])
                fqi_loss.append(results[cat+str(i)]["fqi_losses"])
                
                cal_rewards.append(results[cat+str(i)]["cal_rewards"])
                cal_rewards_running.append(results[cat+str(i)]["cal_rewards_cumulate"])
                cal_loss.append(results[cat+str(i)]["cal_losses"])
                
                cql_rewards.append(results[cat+str(i)]["cql_rewards"])
                cql_rewards_running.append(results[cat+str(i)]["cql_rewards_cumulate"])
                cql_loss.append(results[cat+str(i)]["cql_losses"])
                
                for p_ratio in p_ratios:
                    pescal_rewards[p_ratio].append(results[cat+str(i)]["pescal_rewards"+str(p_ratio)])
                    pescal_rewards_running[p_ratio].append(results[cat+str(i)]["pescal_rewards_cumulate"+str(p_ratio)])
                
                
                
            #fqi
            fqi_rewards_pd=pd.DataFrame(list(map(list, zip(*fqi_rewards))))  
            fqi_rewards_pd["idx"]=np.arange(1,201)
            fqi_rewards_pd = fqi_rewards_pd.melt('idx', var_name='expe', value_name='return')
            
            fqi_rewards_running_pd=pd.DataFrame(list(map(list, zip(*fqi_rewards_running))))
            fqi_rewards_running_pd["idx"]=np.arange(1,201)
            fqi_rewards_running_pd = fqi_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
            
            fqi_loss_pd=pd.DataFrame(list(map(list, zip(*fqi_loss))))  
            fqi_loss_pd["idx"]=np.arange(1,201)
            fqi_loss_pd = fqi_loss_pd.melt('idx', var_name='expe', value_name='return')
            
            #cal
            cal_rewards_pd=pd.DataFrame(list(map(list, zip(*cal_rewards))))  
            cal_rewards_pd["idx"]=np.arange(1,201)
            cal_rewards_pd = cal_rewards_pd.melt('idx', var_name='expe', value_name='return')
            
            cal_rewards_running_pd=pd.DataFrame(list(map(list, zip(*cal_rewards_running))))
            cal_rewards_running_pd["idx"]=np.arange(1,201)
            cal_rewards_running_pd = cal_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
            
            cal_loss_pd=pd.DataFrame(list(map(list, zip(*cal_loss))))  
            cal_loss_pd["idx"]=np.arange(1,201)
            cal_loss_pd = cal_loss_pd.melt('idx', var_name='expe', value_name='return')
            
            #cql
            cql_rewards_pd=pd.DataFrame(list(map(list, zip(*cql_rewards))))  
            cql_rewards_pd["idx"]=np.arange(1,201)
            cql_rewards_pd = cql_rewards_pd.melt('idx', var_name='expe', value_name='return')
            
            cql_rewards_running_pd=pd.DataFrame(list(map(list, zip(*cql_rewards_running))))
            cql_rewards_running_pd["idx"]=np.arange(1,201)
            cql_rewards_running_pd = cql_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
            
            cql_loss_pd=pd.DataFrame(list(map(list, zip(*cql_loss))))  
            cql_loss_pd["idx"]=np.arange(1,201)
            cql_loss_pd = cql_loss_pd.melt('idx', var_name='expe', value_name='return')
            
            #pescal
            pescal_rewards_pd, pescal_rewards_running_pd={},{}
            
            for p_ratio in p_ratios:
                pescal_rewards_pd[p_ratio]=pd.DataFrame(list(map(list, zip(*pescal_rewards[p_ratio]))))  
                pescal_rewards_pd[p_ratio]["idx"]=np.arange(1,201)
                pescal_rewards_pd[p_ratio] = pescal_rewards_pd[p_ratio].melt('idx', var_name='expe', value_name='return')
                
                pescal_rewards_running_pd[p_ratio]=pd.DataFrame(list(map(list, zip(*pescal_rewards_running[p_ratio]))))
                pescal_rewards_running_pd[p_ratio]["idx"]=np.arange(1,201)
                pescal_rewards_running_pd[p_ratio] = pescal_rewards_running_pd[p_ratio].melt('idx', var_name='expe', value_name='return')
                
              
            
            sns.set_theme()
            plt.figure()
            sns.lineplot(data=fqi_rewards_running_pd, x="idx",y="return", label="FQI")
            sns.lineplot(data=cql_rewards_running_pd, x="idx",y="return", label="CQL")
            sns.lineplot(data=cal_rewards_running_pd, x="idx",y="return", label="CAL")
            for p_ratio in p_ratios:
                sns.lineplot(data=pescal_rewards_running_pd[p_ratio], x="idx",y="return", label="PESCAL")#FAVI+pessm"+str(p_ratio)+'sd')
            plt.xlabel("Training Iterations")
            plt.ylabel("online average return")
            pdf.savefig()
        pdf.close()  
  
    

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('-t', '--types', nargs='+', default=[''])
    args = parser.parse_args()
    read(args.iters, args.ratio, args.types)
