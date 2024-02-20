import json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from collections import defaultdict
p_ratios=[1.96]

ratios=[0.0003, 0.5, 1]
types = ["noc"]


def read(iters, ratio, typ, Nodefaultcat=False):
    if not Nodefaultcat and typ:
        c=['']+typ
    else: pass

    exp="pescal_sim"
    for category in c:
        if category:
            filename=f"pescal_sim_{category}_{ratio}"
        else:
            filename=f"pescal_sim_{ratio}"
        
        results={}
        for i in range(iters):
            i+=1
            if category:
                with open("./data/pescal_sim_"+category+'_'+str(ratio)+"_"+str(i)+".json", 'r') as f:
                     results['pescal_sim'+str(i)]= json.load(f)
            else:
                with open("./data/pescal_sim_"+str(ratio)+"_"+str(i)+".json", 'r') as f:
                     results['pescal_sim'+str(i)]= json.load(f)
                

        fqi_rewards = []
        cql_rewards = []
        cal_rewards =[]
        pescal_rewards = defaultdict(list)
        
        
        for i in range(iters):
            i+=1
            fqi_rewards.append(results[exp+str(i)]["fqi_rewards_cumulate"])
            cql_rewards.append(results[exp+str(i)]["cql_rewards_cumulate"])
            
            cal_rewards.append(results[exp+str(i)]["cal_rewards_cumulate"])
            for p_ratio in p_ratios:
                pescal_rewards[p_ratio].append(results[exp+str(i)]["pescal_rewards_cumulate"+str(p_ratio)])
                
        fqi_rewards = np.array(fqi_rewards)
        cql_rewards = np.array(cql_rewards)
        cal_rewards = np.array(cal_rewards)
        for p_ratio in p_ratios:
            pescal_rewards[p_ratio] = np.array(pescal_rewards[p_ratio])
        
        sns.set_theme()
        plt.figure()
        x=(np.arange(len(fqi_rewards[0]))+1)*50
        
        #fqi
        fqi_rewards_mean=fqi_rewards.mean(axis=0)
        fqi_rewards_std=fqi_rewards.std(axis=0)
        plt.plot(x, fqi_rewards_mean, label="FQI", linewidth=3)
        plt.fill_between(x, fqi_rewards_mean-fqi_rewards_std, fqi_rewards_mean+fqi_rewards_std, alpha=0.2)
        
        #cql
        cql_rewards_mean=cql_rewards.mean(axis=0)
        cql_rewards_std=cql_rewards.std(axis=0)
        plt.plot(x, cql_rewards_mean, label="CQL", linewidth=3)
        plt.fill_between(x, cql_rewards_mean-cql_rewards_std, cql_rewards_mean+cql_rewards_std, alpha=0.2)
        
        #cal
        cal_rewards_mean=cal_rewards.mean(axis=0)
        cal_rewards_std=cal_rewards.std(axis=0)
        plt.plot(x, cal_rewards_mean, label="CAL", linewidth=3)
        plt.fill_between(x, cal_rewards_mean-cal_rewards_std, cal_rewards_mean+cal_rewards_std, alpha=0.2)
        
        pescal_rewards_mean={}
        pescal_rewards_std={}
        for p_ratio in p_ratios:
            pescal_rewards_mean[p_ratio]=pescal_rewards[p_ratio].mean(axis=0)
            pescal_rewards_std[p_ratio]=pescal_rewards[p_ratio].std(axis=0)
            plt.plot(x, pescal_rewards_mean[p_ratio], label="PESCAL", linewidth=4)
            plt.fill_between(x, pescal_rewards_mean[p_ratio]-pescal_rewards_std[p_ratio], pescal_rewards_mean[p_ratio]+pescal_rewards_std[p_ratio], alpha=0.2)
        
        if category:#non confounded
            plt.ylim(37.5,57)
        else: #confounded
            plt.ylim(24.5,38.5)
        plt.legend(fontsize=13,loc = 4)
        plt.xlabel("Training iterations")
        plt.ylabel("Averaged discounted return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{filename}.pdf", dpi=1200)
        
        
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--keeping_ratio', type=float)
    parser.add_argument('--type', nargs='+', default=[''])
    args = parser.parse_args()
    for ratio in ratios:
        args.keeping_ratio = float(ratio)
        for t in types:
            args.type = [t]
            read(args.seeds, args.keeping_ratio, args.type)
