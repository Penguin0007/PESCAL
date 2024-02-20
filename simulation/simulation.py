import numpy as np, pandas as pd, torch, torch.nn as nn, json, os, time
from scipy.special import expit
if not os.path.exists('data'):
   os.makedirs('data', exist_ok=True)
device=torch.device("cpu")  # We found the code more cpu-demanding.


def train(env_setting, seed, ratio, rwindow, num_itrs, project_steps, batch_size, discount):
    start_time = time.time()
    print("---------------------------------------")
    print(f"Environment: toy, {env_setting}; Keeping ratio: {ratio}; Seed: {seed}")
    print("---------------------------------------")
    torch.manual_seed(seed)
    np.random.seed(seed)
    p_ratios=[1.96]
    if env_setting == "unconfounded":
        filename = f"pescal_sim_noc_{ratio}_{seed}"
        class Env(object):
            def __init__(self):
                self.num_states=2
                self.num_actions=3
                self.num_mediators=2
                self.num_confounders=2
                
                self.state_space=[0,1]
                self.confounder_space=[-1,1]
                self.action_space=[-1,0,1]
                self.mediator_space=[0,1]
                self.reward_space=[-1,1]
            
            def init_state(self):
                init_state = np.random.binomial(n=1, p=0.5, size=1).item()
                return init_state
            
            def confounder(self, state):
                pc = expit(0.1*state)
                confounder = np.random.choice(self.confounder_space, 1, p=[1-pc, pc]).item()
                return confounder
            
            def action(self, state, confounder):
                pa = expit(state)
                a = np.random.choice(self.action_space, 1, p=[0.5*pa, 1-pa, 0.5*pa]).item()
                return a
            
            def online_action(self, state, pa0, pa1):
                if state==0:
                    a = np.random.choice(self.action_space, 1, p=pa0).item()
                elif state==1:
                    a = np.random.choice(self.action_space, 1, p=pa1).item()
                return a
         
            def mediator(self, state, action):
                pm = expit(0.1*state + action)
                m = np.random.choice(list(reversed(self.mediator_space)), 1, p=[1-pm, pm]).item()
                return m
            
            def reward(self, state, mediator, confounder):
                pr = expit(0.1*state + 2*mediator)
                reward = np.random.choice(self.reward_space, 1, p=[1-pr, pr]).item()
                return reward
         
            def next_state(self, state, mediator, confounder):
                ps = expit(0.1*state + 2*mediator)
                next_state = np.random.binomial(n=1, p=ps, size=1).item()
                return next_state
    else:
        filename = f"pescal_sim_{ratio}_{seed}"
        class Env(object):
            def __init__(self):
                self.num_states=2
                self.num_actions=3
                self.num_mediators=2
                self.num_confounders=2
                
                self.state_space=[0,1]
                self.confounder_space=[-1,1]
                self.action_space=[-1,0,1]
                self.mediator_space=[0,1]
                self.reward_space=[-1,1]
            
            def init_state(self):
                init_state = np.random.binomial(n=1, p=0.5, size=1).item()
                return init_state
            
            def confounder(self, state):
                pc = expit(0.1*state)
                confounder = np.random.choice(self.confounder_space, 1, p=[1-pc, pc]).item()
                return confounder
            
            def action(self, state, confounder):
                pa = expit(state + 2*confounder)
                a = np.random.choice(self.action_space, 1, p=[0.5*pa, 1-pa, 0.5*pa]).item()
                return a
            
            def online_action(self, state, pa0, pa1):
                if state==0:
                    a = np.random.choice(self.action_space, 1, p=pa0).item()
                elif state==1:
                    a = np.random.choice(self.action_space, 1, p=pa1).item()
                return a
         
            def mediator(self, state, action):
                pm = expit(0.1*state + action)
                m = np.random.choice(list(reversed(self.mediator_space)), 1, p=[1-pm, pm]).item()
                return m
            
            def reward(self, state, mediator, confounder):
                pr = expit(2*confounder + 0.1*state + 2*mediator)
                reward = np.random.choice(self.reward_space, 1, p=[1-pr, pr]).item()
                return reward
         
            def next_state(self, state, mediator, confounder):
                ps = expit(2*confounder + 0.1*state + 2*mediator)
                next_state = np.random.binomial(n=1, p=ps, size=1).item()
                return next_state

    env=Env()

    #Emperical optimal policy
    def online_reward(env, size, discount):
        emperical_reward={(a0,a1):0 for a0 in env.action_space for a1 in env.action_space}
        det_policy=np.identity(3)
        s=env.init_state()
        for i in range(env.num_actions):
            a0=env.action_space[i]
            pa0=det_policy[i]
            for j in range(env.num_actions):
                a1=env.action_space[j]
                pa1=det_policy[j]
                rewards=0
                for idx in range(size):
                    c=env.confounder(s)
                    a=env.online_action(s,pa0,pa1)
                    m=env.mediator(s, a)
                    r=env.reward(s, m, c)
                    s_prime=env.next_state(s, m, c)
                    s=s_prime
                    rewards += discount**(idx+1)*r
                emperical_reward[(a0,a1)]=rewards
        return emperical_reward

    def average_emperical_reward(env, discount, trajectory, horizon):
        emperical_reward={(a0,a1):0 for a0 in env.action_space for a1 in env.action_space}
        for i in range(trajectory):
            online_reward_single=online_reward(env, horizon, discount)
            for a1 in env.action_space:
                for a2 in env.action_space:
                    emperical_reward[(a1,a2)]+=online_reward_single[(a1,a2)]
        emperical_reward.update((x,y/trajectory) for x,y in emperical_reward.items())
        return emperical_reward
    
    # print("Online environment observations...")
    # emperical_reward=average_emperical_reward(env, discount, trajectory=100, horizon=1000)            
    # opt_pol=max(emperical_reward, key=emperical_reward.get)
    # print("Online reward for all possible policy (state 0, state 1):\n", emperical_reward)
    # print("Online optimal policy for state 0:", opt_pol[0], " state 1: ", opt_pol[1])


    ###Generate offline training dataset
    def generate_training_dataset(env, epoch, horizon):
        training_dataset=[]
        for _ in range(epoch):
            s=env.init_state()
            for _ in range(horizon):
                c=env.confounder(s)
                a=env.action(s,c)
                m=env.mediator(s, a)
                r=env.reward(s, m, c)
                s_prime=env.next_state(s, m, c)
                training_dataset.append((s,a,m,r,s_prime))
                s=s_prime
        return training_dataset
    
    print("\nGenerating offline dataset...")
    training_dataset_orig=generate_training_dataset(env, epoch=100, horizon=500)
    
    training_dataset=training_dataset_orig[:round(ratio*len(training_dataset_orig))]
    for e in training_dataset_orig[round(ratio*len(training_dataset_orig)):]:
        if not (e[1]==0 or e[1]==1):
            training_dataset.append(e)
        else:
            pass
        
    print("\nCalculating front-door adjustment probabilities...")
    ###Front-door adjustment    
    #Goal P(r|s,do(a))=\sum P(m|s,a)P(r|s,a*,m)P(a*|s)  &  P(s'|s,do(a))=\sum P(m|s,a)P(s'|s,a*,m)P(a*|s)
    pd_training_data=pd.DataFrame(training_dataset)
    pd_training_data.columns=["s","a","m","r","s'"]
    n=len(training_dataset)

    #P(a|s)
    Pa_s={(s,a): np.mean(pd_training_data.loc[lambda pd_training_data: pd_training_data['s']==s]["a"].values==a)
          for s in env.state_space
          for a in env.action_space}
    for key,value in Pa_s.items():
        if np.isnan(value):
            Pa_s[key]=0

    #P(m|s,a)
    Pm_sa={(s,a,m): np.mean(pd_training_data.loc[lambda pd_training_data: pd_training_data['s']==s]
                            .loc[lambda pd_training_data: pd_training_data['a']==a]['m'].values==m)
           for s in env.state_space
           for a in env.action_space
           for m in env.mediator_space}
    for key,value in Pm_sa.items():
        if np.isnan(value):
            Pm_sa[key]=1/(10*n)
    
    #Penalize over P(m|s,a)
    p_ucb={(s,a):np.sqrt(2*np.log(n)/max(len(pd_training_data[(pd_training_data['s']==s)&(pd_training_data['a']==a)]), 0.01))
            for s in env.state_space
            for a in env.action_space}
    for key,value in p_ucb.items():
        if np.isnan(value):
            p_ucb[key]=0
    
    sd_psam={(s,a,m):(np.sqrt(Pm_sa[(s,a,m)]*(1-Pm_sa[(s,a,m)])/max(len(pd_training_data[(pd_training_data['s']==s)&(pd_training_data['a']==a)]), 1/(1e4*n))) 
                         if len(pd_training_data[(pd_training_data['s']==s)&(pd_training_data['a']==a)])>=30
                         else np.sqrt(0.5*(1-0.5)/max(len(pd_training_data[(pd_training_data['s']==s)&(pd_training_data['a']==a)]), 1/(1e4*n))))
            for s in env.state_space
            for a in env.action_space
            for m in env.mediator_space}

    #P(r|s,a,m)
    Pr_sam={(s,a,m,r): np.mean(pd_training_data.loc[lambda pd_training_data: pd_training_data['s']==s]
                               .loc[lambda pd_training_data: pd_training_data['a']==a]
                               .loc[lambda pd_training_data: pd_training_data['m']==m]['r'].values==r)
            for s in env.state_space
            for a in env.action_space
            for m in env.mediator_space
            for r in env.reward_space}
    for key,value in Pr_sam.items():
        if np.isnan(value):
            Pr_sam[key]=0

    #P(s'|s,a,m)
    Psprime_sam={(s,a,m,s_prime): np.mean(pd_training_data.loc[lambda pd_training_data: pd_training_data['s']==s]
                                          .loc[lambda pd_training_data: pd_training_data['a']==a]
                                          .loc[lambda pd_training_data: pd_training_data['m']==m]["s'"].values==s_prime)
            for s in env.state_space
            for a in env.action_space
            for m in env.mediator_space
            for s_prime in env.state_space}
    for key,value in Psprime_sam.items():
        if np.isnan(value):
            Psprime_sam[key]=0              
                                                        
    def recover_q_sa(env, mqvalues, penalize=False, penalize_m=False, pm_method=None, p_ratio=None):
        dS=env.num_states
        dA=env.num_actions
        dM=env.num_mediators
        qvalues = np.zeros((dS, dA))
        for s_idx in range(dS):
            s=env.state_space[s_idx]
            for a_idx in range(dA):
                a=env.action_space[a_idx]
                qvalue=0
                for m_idx in range(dM):
                    m=env.mediator_space[m_idx]
                    for a_tilde_idx in range(dA):
                        a_tilde=env.action_space[a_tilde_idx]
                        if penalize_m:
                            # if pm_method=='ucb':
                            #     qvalue+=mqvalues[s_idx,a_tilde_idx,m_idx]*Pa_s[(s,a_tilde)]*(Pm_sa[(s,a,m)]-pm_ucb[(s,a,m)])
                            # if pm_method=='belle':
                            #     qvalue+=mqvalues[s_idx,a_tilde_idx,m_idx]*Pa_s[(s,a_tilde)]*(Pm_sa[(s,a,m)]-pm_belle[(s,a,m)])
                            if pm_method=='sd':
                                qvalue+=mqvalues[s_idx,a_tilde_idx,m_idx]*Pa_s[(s,a_tilde)]*(Pm_sa[(s,a,m)]-p_ratio*sd_psam[(s,a,m)])
                        else:
                            qvalue+=mqvalues[s_idx,a_tilde_idx,m_idx]*Pa_s[(s,a_tilde)]*Pm_sa[(s,a,m)]
                if penalize:
                    qvalues[s_idx, a_idx]=qvalue-p_ucb[(s,a)]
                else:
                    qvalues[s_idx, a_idx]=qvalue
        return qvalues
    

    ##################Algorithms######################
    print("\nBegin training with different algorithms...")
    np_training_data=np.array(training_dataset)

    #Neural Networks
    class Net(nn.Module):
        def __init__(self, env):
            super(Net, self).__init__()
            self.fc_seqn = nn.Sequential(
                nn.Linear(env.num_states, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, env.num_actions))
        def forward(self, s):
            s = self.fc_seqn(s)
            return s
        
    #Utility functions
    def onestep_online_reward(env, policy, epoch=10, horizon=500, discount=0.99):
        rewards_list = []
        for _ in range(epoch):
            reward = 0
            s=env.init_state()
            for idx in range(horizon):
                a=policy[s]
                c=env.confounder(s)
                m=env.mediator(s, a)
                r=env.reward(s, m, c)
                s_prime=env.next_state(s, m, c)
                s=s_prime
                reward += discount**(idx+1)*r
            rewards_list.append(reward)
        return np.mean(rewards_list)

    def one_hot(y, n_dims=None):
        y_tensor = y.view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot
        
    def q_backup_sampled(env, q_values, r, s_prime_idx, discount):
        q_values_sprime = q_values[s_prime_idx, :]
        values = np.max(q_values_sprime, axis=-1)
        target_value = r + discount * values
        return target_value
    
    def project_qvalues_sampled(s_idx, a_idx, target_values, network, optimizer):
        target_qvalues = torch.tensor(target_values, dtype=torch.float32).to(device)
        s_idx = torch.tensor(s_idx, dtype=torch.int64)
        s_idx_onehot = one_hot(s_idx,env.num_states).to(device)
        a_idx = torch.tensor(a_idx, dtype=torch.int64).to(device)
        pred_qvalues = network(s_idx_onehot)
        pred_qvalues = pred_qvalues.gather(1, a_idx.reshape(-1,1)).squeeze()
        loss = torch.mean((pred_qvalues - target_qvalues)**2)
        network.zero_grad()
        loss.backward()
        optimizer.step()

        s_onehot = one_hot(torch.arange(env.num_states), env.num_states).to(device)
        pred_qvalues = network(s_onehot)
        return pred_qvalues.detach().cpu().numpy()  
    
    ############################################FQI################################################
    def FQI(env, net, batch_size, num_itrs, discount, project_steps, training_dataset=None):
        dS = env.num_states
        dA = env.num_actions
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        q_values = np.zeros((dS, dA))
        qvalue_list=[]
        online_emperical=[]
        online_reward=[]
        for i in range(num_itrs):
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            data=training_dataset[training_idx]
            s, a, m, r, s_prime= data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
            s_idx, a_idx, m_idx, s_prime_idx = \
            np.array([env.state_space.index(s_i) for s_i in s]), \
            np.array([env.action_space.index(a_i) for a_i in a]), \
            np.array([env.mediator_space.index(m_i) for m_i in m]), \
            np.array([env.state_space.index(s_prime_i) for s_prime_i in s_prime])
                
            target_values = q_backup_sampled(env, q_values, r, s_prime_idx, discount)
            intermed_values = project_qvalues_sampled(s_idx, a_idx, target_values, net, optimizer)
            if (i+1) % project_steps == 0:
                q_values = intermed_values
                qvalue_list.append(q_values)
                policy=[env.action_space[i] for i in np.argmax(q_values, axis=1)] 
                online_emperical.append(onestep_online_reward(env, policy))
                l=len(online_emperical)
                if l<=rwindow:
                    online_reward.append(np.mean(online_emperical))
                else:
                    online_reward.append(np.mean(online_emperical[-rwindow:]))
        return q_values, qvalue_list, np.mean(online_emperical), online_emperical, online_reward

    net = Net(env)
    fqi_qvalues, fqi_qvalues_list, fqi_online, fqi_reward_list, fqi_reward_cumulate = FQI(env, net, batch_size, num_itrs, discount, project_steps, training_dataset=np_training_data)
    fqi_policy=[env.action_space[i] for i in np.argmax(fqi_qvalues, axis=1)]     
    print("\nTraining results for FQI...")
    print("Optimal q values from FQI: \n",fqi_qvalues)
    print("Optimal policy for FQI:\nstate 0: action", fqi_policy[0], ", state 1: action", fqi_policy[1])
    print("FQI online mean reward (std) with number of iterations %d and number of projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, fqi_online, np.std(fqi_reward_list)))
        
    ############################################CQL################################################
    cql_alpha=0.1
    #Utility functions
    def project_qvalues_cql(s_idx, a_idx, target_values, network, optimizer, cql_alpha):
        target_qvalues = torch.tensor(target_values, dtype=torch.float32).to(device)
        s_idx = torch.tensor(s_idx, dtype=torch.int64)
        s_idx_onehot = one_hot(s_idx,env.num_states).to(device)
        a_idx = torch.tensor(a_idx, dtype=torch.int64).to(device)
        pred_qvalues = network(s_idx_onehot)
        logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)
        
        pred_qvalues = pred_qvalues.gather(1, a_idx.reshape(-1,1)).squeeze()
        cql_loss = logsumexp_qvalues - pred_qvalues
        loss = torch.mean((pred_qvalues - target_qvalues)**2) + cql_alpha * torch.mean(cql_loss)
        network.zero_grad()
        loss.backward()
        optimizer.step()

        s_onehot = one_hot(torch.arange(env.num_states), env.num_states).to(device)
        pred_qvalues = network(s_onehot)
        return pred_qvalues.detach().cpu().numpy()
    
    def CQL(env, net, batch_size, num_itrs, discount, project_steps, cql_alpha, training_dataset=None):
        dS = env.num_states
        dA = env.num_actions
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        q_values = np.zeros((dS, dA))
        qvalue_list=[]
        online_emperical=[]
        online_reward=[]
        for i in range(num_itrs):
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            data=training_dataset[training_idx]
            s, a, m, r, s_prime= data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
            s_idx, a_idx, m_idx, s_prime_idx = \
            np.array([env.state_space.index(s_i) for s_i in s]), \
            np.array([env.action_space.index(a_i) for a_i in a]), \
            np.array([env.mediator_space.index(m_i) for m_i in m]), \
            np.array([env.state_space.index(s_prime_i) for s_prime_i in s_prime])
                
            target_values = q_backup_sampled(env, q_values, r, s_prime_idx, discount)
            intermed_values = project_qvalues_cql(s_idx, a_idx, target_values, net, optimizer,  cql_alpha)
            if (i+1) % project_steps == 0:
                q_values = intermed_values
                qvalue_list.append(q_values)
                policy=[env.action_space[i] for i in np.argmax(q_values, axis=1)] 
                online_emperical.append(onestep_online_reward(env, policy))
                l=len(online_emperical)
                if l<=rwindow:
                    online_reward.append(np.mean(online_emperical))
                else:
                    online_reward.append(np.mean(online_emperical[-rwindow:]))
        return q_values, qvalue_list, np.mean(online_emperical), online_emperical, online_reward

    net = Net(env)
    cql_qvalues, cql_qvalues_list, cql_online, cql_reward_list, cql_reward_cumulate = CQL(env, net, batch_size, num_itrs, discount,\
                                                                                          project_steps, cql_alpha, training_dataset=np_training_data)
    cql_policy=[env.action_space[i] for i in np.argmax(cql_qvalues, axis=1)]  
    print("\nTraining results for CQL...")
    print("Optimal q values from CQL: \n",cql_qvalues)
    print("Optimal policy for CQL:\nstate 0: action", cql_policy[0], ", state 1: action", cql_policy[1])
    print("CQL online mean reward (std) with number of iterations %d and number of projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, cql_online, np.std(cql_reward_list)))


    ###############################################################################################
    #######################################CAL & PESCAL############################################
    ###############################################################################################
    #Neural net
    class mNet(nn.Module):
        def __init__(self, env):
            super(mNet, self).__init__()
            self.fc_seqn = nn.Sequential(
                nn.Linear(env.num_states, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, env.num_actions*env.num_mediators))
        def forward(self, s):
            s = self.fc_seqn(s)
            s = s.view(-1,env.num_actions,env.num_mediators)
            return s

    #Utility functions
    def mq_backup_sampled(env, mq_values, r, s_prime_idx, penalize=False, penalize_m=False, pm_method=None, p_ratio=None, discount=0.99):
        dA = env.num_actions
        dM = env.num_mediators
        s_prime_full=[env.state_space[i] for i in list(s_prime_idx)]
        
        interm_aprime=np.zeros((len(s_prime_full),dA))
        for i in range(len(s_prime_full)):
            s_prime=s_prime_full[i]
            for a_prime_idx in range(dA):
                a_prime=env.action_space[a_prime_idx]
                interm=0
                for m_prime_idx in range(dM):
                    m_prime=env.mediator_space[m_prime_idx]
                    for a_tilde_prime_idx in range(dA):
                        a_tilde_prime=env.action_space[a_tilde_prime_idx]
                        if penalize_m:
                            # if pm_method=="ucb":
                            #     interm += mq_values[s_prime_idx[i],a_tilde_prime_idx,m_prime_idx]\
                            #     *Pa_s[(s_prime,a_tilde_prime)]*(Pm_sa[(s_prime,a_prime,m_prime)]-pm_ucb[(s_prime,a_prime,m_prime)])
                            if pm_method=="sd":
                                interm += mq_values[s_prime_idx[i],a_tilde_prime_idx,m_prime_idx]\
                                *Pa_s[(s_prime,a_tilde_prime)]*(Pm_sa[(s_prime,a_prime,m_prime)]-p_ratio*sd_psam[(s_prime,a_prime,m_prime)])
                        else:
                            interm += mq_values[s_prime_idx[i],a_tilde_prime_idx,m_prime_idx]\
                            *Pa_s[(s_prime,a_tilde_prime)]*Pm_sa[(s_prime,a_prime,m_prime)]
                if penalize:
                    interm_aprime[i,a_prime_idx]=interm-p_ucb[(s_prime,a_prime)]
                else:
                    interm_aprime[i,a_prime_idx]=interm 
        target_value = r + discount * np.max(interm_aprime, axis=1)
        return target_value

    def project_mqvalues_sampled(s_idx, a_idx, m_idx, target_values, network, optimizer):
        target_qvalues = torch.tensor(target_values, dtype=torch.float32).to(device)
        s_idx = torch.tensor(s_idx, dtype=torch.int64)
        s_idx_onehot = one_hot(s_idx, env.num_states).to(device)
        a_idx = torch.tensor(a_idx, dtype=torch.int64).to(device)
        m_idx = torch.tensor(m_idx, dtype=torch.int64).to(device)
        pred_qvalues = network(s_idx_onehot)
        
        pred_qvalues = pred_qvalues[torch.tensor(list(range(len(s_idx)))).to(device), a_idx, m_idx]
        loss = torch.mean((pred_qvalues - target_qvalues)**2)
        network.zero_grad()
        loss.backward()
        optimizer.step()

        s_onehot = one_hot(torch.arange(env.num_states), env.num_states).to(device)
        pred_qvalues = network(s_onehot)
        return pred_qvalues.detach().cpu().numpy()  
      
    def PESCAL(env, net, batch_size, num_itrs, penalize=False, penalize_m=False, pm_method=None, p_ratio=None,  discount=0.99, project_steps=50, training_dataset=None):
        dS = env.num_states
        dA = env.num_actions
        dM = env.num_mediators
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        mq_values = np.zeros((dS, dA, dM))
        recover_fqi_qvalues_list=[]
        online_emperical=[]
        online_reward=[]
        for i in range(num_itrs):
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            data=training_dataset[training_idx]
            s, a, m, r, s_prime= data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
            s_idx, a_idx, m_idx, s_prime_idx = \
            np.array([env.state_space.index(s_i) for s_i in s]), \
            np.array([env.action_space.index(a_i) for a_i in a]), \
            np.array([env.mediator_space.index(m_i) for m_i in m]), \
            np.array([env.state_space.index(s_prime_i) for s_prime_i in s_prime])

            
            target_values = mq_backup_sampled(env, mq_values, r, s_prime_idx, penalize)
            intermed_values = project_mqvalues_sampled(s_idx, a_idx, m_idx, target_values, net, optimizer)
        
            if (i+1) % project_steps == 0:
                mq_values = intermed_values
                qvalue = recover_q_sa(env, mq_values-np.min(mq_values), penalize, penalize_m, pm_method,p_ratio)
                recover_fqi_qvalues_list.append(qvalue)
                recover_fqi_policy=[env.action_space[i] for i in np.argmax(qvalue, axis=1)]
                online_emperical.append(onestep_online_reward(env, recover_fqi_policy))
                l=len(online_emperical)
                if l<=rwindow:
                    online_reward.append(np.mean(online_emperical))
                else:
                    online_reward.append(np.mean(online_emperical[-rwindow:]))
        return qvalue, recover_fqi_qvalues_list, np.mean(online_emperical), online_emperical, online_reward
    
    #CAL
    mnet = mNet(env)
    cal_qvalues, cal_qvalues_list, cal_online, cal_reward_list, cal_reward_cumulate = PESCAL(env, net=mnet, batch_size=batch_size, num_itrs=num_itrs, project_steps=project_steps, training_dataset=np_training_data)
    cal_policy=[env.action_space[i] for i in np.argmax(cal_qvalues, axis=1)]     
    print("\nTraining results for CAL...")
    print("Optimal q values from CAL: \n",cal_qvalues)
    print("Optimal policy for CAL:\nstate 0: action", cal_policy[0], ", state 1: action", cal_policy[1])
    print("CAL online mean reward (std) with number of iterations %d and number of projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, cal_online, np.std(cal_reward_list)))
    
    #PESCAL
    pescal_reward_list={}
    pescal_reward_cumulate={}
    for p_ratio in p_ratios:
        mnet = mNet(env)
        pescal_qvalues, pescal_qvalues_list, pescal_online, pescal_reward_list[str(p_ratio)], pescal_reward_cumulate[str(p_ratio)] = PESCAL(env, mnet, batch_size, num_itrs, penalize_m=True, pm_method='sd', p_ratio=p_ratio, project_steps=project_steps, training_dataset=np_training_data)
        pescal_policy=[env.action_space[i] for i in np.argmax(pescal_qvalues, axis=1)]  
        print("\nTraining results for PESCAL...")
        print("Optimal q values from PESCAL: \n",pescal_qvalues)
        print("Optimal policy for PESCAL:\nstate 0: action", pescal_policy[0], ", state 1: action", pescal_policy[1])
        print("PESCAL online mean reward (std) with number of iterations %d and number of projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, pescal_online, np.std(pescal_reward_list[str(p_ratio)])))

    dictionary = {
        "fqi_rewards": fqi_reward_list,
        "fqi_rewards_cumulate": fqi_reward_cumulate,
        "cql_rewards": cql_reward_list,
        "cql_rewards_cumulate": cql_reward_cumulate,
        "cal_rewards": cal_reward_list,
        "cal_rewards_cumulate": cal_reward_cumulate}
    
    for p_ratio in p_ratios:
        dictionary["pescal_rewards"+str(p_ratio)]=pescal_reward_list[str(p_ratio)]
        dictionary["pescal_rewards_cumulate"+str(p_ratio)]=pescal_reward_cumulate[str(p_ratio)]


    with open(f"./data/{filename}.json", "w") as outfile:
        json.dump(dictionary, outfile)
        
    end_time = time.time()
    print('\nTakes time: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_setting', default="confounded")              # Environment is confounded or unconfounded
    parser.add_argument('--seed', type=int)                                 # Seed for PyTorch and Numpy, we use 100 seeds in total
    parser.add_argument('--keeping_ratio', type=float)                      # Ratio of the original data (50000 observation tuples in total) to keep
    parser.add_argument('--rwindow', type=int, default=50)                  # Moving window length of which we take average of emperical online reward
    parser.add_argument('--training_steps', type=int, default=10000)        # Total training steps
    parser.add_argument('--project_steps', type=float, default=50)          # How often in steps do we evaluate
    parser.add_argument('--batch_size', type=int, default=128)              # Batch size to sample from offline dataset during training
    parser.add_argument('--discount', type=float, default=0.99)             # Discount factor
    args = parser.parse_args()
    
    env_settings = ["unconfounded", "confounded"]
    num_seeds = list(range(100))
    keeping_ratio_list = [0.0003, 0.5, 1]
    
    for e in env_settings:
        args.env_setting = e
        for i in num_seeds:
            args.seed = i
            for kr in keeping_ratio_list:
                args.keeping_ratio = float(kr)
                train(args.env_setting, args.seed, args.keeping_ratio, args.rwindow, 
                      args.training_steps, args.project_steps, args.batch_size, args.discount)
    
