import json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
p_ratios=[1.96]

def read(iters, ratio, c, Nodefaultcat=False):
    if not Nodefaultcat and c:
        c=['']+c
    else: pass
    exp="pescal_sim"
    for category in c:
        if category:
            pdf=PdfPages("pescal_sim_"+category+'_'+str(ratio)+".pdf")
        else:
            pdf=PdfPages("pescal_sim_"+str(ratio)+".pdf")
        
        results={}
        for i in range(iters):
            i+=1
            if category:
                with open("./data/pescal_sim_"+category+'_'+str(ratio)+"_"+str(i)+".json", 'r') as f:
                     results['pescal_sim'+str(i)]= json.load(f)
            else:
                with open("./data/pescal_sim_"+str(ratio)+"_"+str(i)+".json", 'r') as f:
                     results['pescal_sim'+str(i)]= json.load(f)
                

        fqi_rewards, fqi_rewards_running=[], []
        cql_rewards, cql_rewards_running=[], []
        cal_rewards, cal_rewards_running=[], []
        pescal_rewards, pescal_rewards_running=defaultdict(list), defaultdict(list)
        
        
        for i in range(iters):
            i+=1
            fqi_rewards.append(results[exp+str(i)]["fqi_rewards"])
            fqi_rewards_running.append(results[exp+str(i)]["fqi_rewards_cumulate"])
            cql_rewards.append(results[exp+str(i)]["cql_rewards"])
            cql_rewards_running.append(results[exp+str(i)]["cql_rewards_cumulate"])
            
            cal_rewards.append(results[exp+str(i)]["cal_rewards"])
            cal_rewards_running.append(results[exp+str(i)]["cal_rewards_cumulate"])
            for p_ratio in p_ratios:
                pescal_rewards[p_ratio].append(results[exp+str(i)]["pescal_rewards"+str(p_ratio)])
                pescal_rewards_running[p_ratio].append(results[exp+str(i)]["pescal_rewards_cumulate"+str(p_ratio)])

        #fqi
        fqi_rewards_pd=pd.DataFrame(list(map(list, zip(*fqi_rewards))))  
        fqi_rewards_pd["idx"]=np.arange(1,201)
        fqi_rewards_pd = fqi_rewards_pd.melt('idx', var_name='expe', value_name='return')
        
        fqi_rewards_running_pd=pd.DataFrame(list(map(list, zip(*fqi_rewards_running))))
        fqi_rewards_running_pd["idx"]=np.arange(1,201)
        fqi_rewards_running_pd = fqi_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
        
        #cql
        cql_rewards_pd=pd.DataFrame(list(map(list, zip(*cql_rewards))))  
        cql_rewards_pd["idx"]=np.arange(1,201)
        cql_rewards_pd = cql_rewards_pd.melt('idx', var_name='expe', value_name='return')
        
        cql_rewards_running_pd=pd.DataFrame(list(map(list, zip(*cql_rewards_running))))
        cql_rewards_running_pd["idx"]=np.arange(1,201)
        cql_rewards_running_pd = cql_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
        
        #cal
        cal_rewards_pd=pd.DataFrame(list(map(list, zip(*cal_rewards))))  
        cal_rewards_pd["idx"]=np.arange(1,201)
        cal_rewards_pd = cal_rewards_pd.melt('idx', var_name='expe', value_name='return')
        
        cal_rewards_running_pd=pd.DataFrame(list(map(list, zip(*cal_rewards_running))))
        cal_rewards_running_pd["idx"]=np.arange(1,201)
        cal_rewards_running_pd = cal_rewards_running_pd.melt('idx', var_name='expe', value_name='return')
        
        #pescal
        pescal_rewards_pd, pescal_rewards_running_pd = {},{}
        for p_ratio in p_ratios:
            pescal_rewards_pd[p_ratio]=pd.DataFrame(list(map(list, zip(*pescal_rewards[p_ratio]))))  
            pescal_rewards_pd[p_ratio]["idx"]=np.arange(1,201)
            pescal_rewards_pd[p_ratio] = pescal_rewards_pd[p_ratio].melt('idx', var_name='expe', value_name='return')
            
            pescal_rewards_running_pd[p_ratio]=pd.DataFrame(list(map(list, zip(*pescal_rewards_running[p_ratio]))))
            pescal_rewards_running_pd[p_ratio]["idx"]=np.arange(1,201)
            pescal_rewards_running_pd[p_ratio] = pescal_rewards_running_pd[p_ratio].melt('idx', var_name='expe', value_name='return')
            
        
        sns.set_theme()
        plt.figure()
        sns.lineplot(data=fqi_rewards_running_pd, x="idx",y="return", label="FQI", linewidth=3)
        sns.lineplot(data=cql_rewards_running_pd, x="idx",y="return", label="CQL", linewidth=3)
        sns.lineplot(data=cal_rewards_running_pd, x="idx",y="return", label="Causal FQI", linewidth=3)

        for p_ratio in p_ratios:
            sns.lineplot(data=pescal_rewards_running_pd[p_ratio], x="idx",y="return", label="PESCAL", linewidth=4)#+str(p_ratio)+'sd')
        
        # if category:#non confounded
        #     plt.ylim(37.5,57)
        # else: #confounded
        #     plt.ylim(24.5,38.5)
        plt.legend(fontsize=13,loc = 4)
        plt.xlabel("Training iterations")
        plt.ylabel("online average return")
        plt.legend()
        pdf.savefig()
        pdf.close()
    

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', type=int, default=100)
    parser.add_argument('-r', '--ratio', type=float)
    parser.add_argument('-c', '--categories', nargs='+', default=[''])
    args = parser.parse_args()
    read(args.iters, args.ratio, args.categories)
