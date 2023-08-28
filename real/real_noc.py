import numpy as np, torch, torch.nn as nn, time, os, statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
device=torch.device("cpu")
if not os.path.exists('data'):
   os.makedirs('data', exist_ok=True)

def train(iters, rwindow, ratio):
    start_time = time.time()
    #ratio=1 rwindow=50
    #MDP
    class Env(object):
        def __init__(self):
            self.confounder_space=[-1,1]
            self.action_space=[0,1]
            self.mediator_space=[0,1]
            
            self.dim_state=3
            self.num_actions=len(self.action_space)
            self.num_confounders=len(self.confounder_space)
            self.num_mediators=len(self.mediator_space)
        
        def init_state(self):
            init_state = np.random.normal(size=self.dim_state, scale=0.1)
            return init_state

        def confounder(self, state):
            confounder = np.random.choice(self.confounder_space, 1, p=[0.5, 0.5])
            return confounder

        def action(self, state, confounder):
            pa = float(expit(0.1*np.sum(state)))
            a = np.random.choice(self.action_space, 1, p=[1-pa, pa])
            return a

        def mediator(self, state, action):
            pm = float(expit(0.1*np.sum(state) + 0.9*(action - 0.5)))
            m = np.random.choice(self.mediator_space, 1, p=[1-pm, pm])
            return m

        def reward(self, state, mediator, confounder, random=True):
            if self.dim_state == 1:
                rmean = 0.5*(state[0] + mediator) - 0.1*state[0]
            elif self.dim_state >= 2:
                rmean = 0.5*(np.sum(state) + mediator) - 0.1*np.sum(state)
                pass
            if random:
                reward = np.random.normal(size=1, loc=rmean, scale=0.1)
            else:
                reward = rmean
            return reward

        def next_state(self, state, mediator, confounder):
            next_state = np.copy(state)
            if self.dim_state >= 2:
                next_state = 0.5*(state + mediator) - 0.1*state
            elif self.dim_state == 1:
                next_state[0] = 0.5*(state[0] + mediator) - 0.1*state[0]
            else:
                pass
            cov_matrix = 1*np.eye(self.dim_state)
            next_state = np.random.multivariate_normal(size=1, mean=next_state, cov=cov_matrix)
            next_state = next_state.flatten()
            return next_state

    env=Env()

    #Emperical optimal policy
    def online_reward(env, size, discount=0.99):
        emperical_reward={a:0 for a in env.action_space}
        s=env.init_state()
        for a in env.action_space:
            rewards=0
            for idx in range(size):
                c=env.confounder(s)
                m=env.mediator(s, a)
                r=env.reward(s, m, c)
                s_prime=env.next_state(s, m, c)
                s=s_prime
                rewards += discount**(idx+1)*r
            emperical_reward[a]=rewards
        return emperical_reward

    def average_emperical_reward(env, trajectory, horizon):
        emperical_reward={a:0 for a in env.action_space}
        for i in range(trajectory):
            online_reward_single=online_reward(env, size=horizon)
            for a in env.action_space:
                emperical_reward[a]+=online_reward_single[a]
        emperical_reward.update((x,y/trajectory) for x,y in emperical_reward.items())
        return emperical_reward

    emperical_reward=average_emperical_reward(env, trajectory=100, horizon=1000)            
    opt_pol=max(emperical_reward, key=emperical_reward.get)
    print("Online reward for all possible policy:\n", emperical_reward)
    print("Online optimal policy", opt_pol)
                                 
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
    
    training_dataset_orig=generate_training_dataset(env, epoch=100, horizon=500)
    
    training_dataset=training_dataset_orig[:round(ratio*len(training_dataset_orig))]
    for e in training_dataset_orig[round(ratio*len(training_dataset_orig)):]:
        if not (e[1]==0):
            training_dataset.append(e)
        else:
            pass

    #Data pre processing
    s=[]
    [s.append(t[0]) for t in training_dataset]
    s=np.array(s)
    a=np.array([t[1] for t in training_dataset]).squeeze()
    a_idx = np.array([env.action_space.index(a_i) for a_i in a])
    m=np.array([t[2] for t in training_dataset]).squeeze()
    r=np.array([t[3] for t in training_dataset]).squeeze()
    sprime=[]
    [sprime.append(t[4]) for t in training_dataset]
    sprime=np.array(sprime)
    
    _, counts =np.unique(a, return_counts=True)

    ###Front-door adjustment    
    #Goal P(r|s,do(a))=\sum P(m|s,a)P(r|s,a*,m)P(a*|s)  &  P(s'|s,do(a))=\sum P(m|s,a)P(s'|s,a*,m)P(a*|s)
    #Get P(a|s) by logistic regression
    Pa_smodel=LogisticRegression().fit(s,a)

    def Pa_s(s,a):
        probs=Pa_smodel.predict_proba(s)
        return probs[np.arange(len(probs)),a]

    Pas_mse=np.mean((1-Pa_smodel.predict_proba(s)[np.arange(len(Pa_smodel.predict_proba(s))), a])**2)
    print(f"Accuracy score for prediction of a|s: {accuracy_score(Pa_smodel.predict(s),a_idx)}")
    print(f"MSE for prediction probability for P(a|s): {Pas_mse}")

    #Learn P(m|s,a)
    #statmodels
    sac=np.c_[np.ones(len(s)),s,a]
    Pm_samodel=sm.Logit(m, sac).fit()
    cov=Pm_samodel.cov_params()
    
    def Pm_sa(s,a,m):
        int_sac=np.c_[np.ones(len(s)),s,a]
        probs=Pm_samodel.predict(int_sac)
        p=np.c_[1-probs,probs]
        return p[np.arange(len(p)),m]
    

    # sPm_samodel.summary()
    def getvar(s,a):
        x=np.c_[np.ones(len(s)),s,a]
        return np.diag(x@cov@x.T)
        
    Pmsa_mse=np.mean((1-np.c_[1-Pm_samodel.predict(sac),Pm_samodel.predict(sac)][np.arange(len(Pm_samodel.predict(sac))), m])**2)
    print(f"Accuracy score for prediction of m|s,a: {accuracy_score(Pm_samodel.predict(sac)>=0.5,m)}")
    print(f"MSE for prediction probability for P(m|s,a): {Pmsa_mse}")
    

    ##################Algorithms######################
    num_itrs=10000
    project_steps=50
    batch_size=128
    tau=0.99
    
    #Neural net
    class qNet(nn.Module):
        def __init__(self, env):
            super(qNet, self).__init__()
            self.fc_seqn = nn.Sequential(
                nn.Linear(env.dim_state, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, env.num_actions))
        def forward(self, s):
            q = self.fc_seqn(s)
            return q
        
    #Utility fcns
    def trajectory_online_reward(env, net, penalize_m=False, mediator=False, epoch=10, horizon=500, discount=0.99):
        rewards_list=[]
        for _ in range(epoch):
            reward=0
            s=env.init_state()
            for idx in range(horizon):
                if mediator:
                    qvalues=recover_q_sa(env, net, torch.tensor(s, dtype=torch.float).reshape(-1,env.dim_state).to(device), penalize_m)
                    a = np.array(env.action_space)[np.argmax(qvalues, axis=-1)]
                else:
                    a=env.action_space[torch.argmax(net(torch.tensor(s, dtype=torch.float).to(device)))]
                c=env.confounder(s)
                m=env.mediator(s, a)
                r=env.reward(s, m, c)
                s_prime=env.next_state(s, m, c)
                s=s_prime
                reward += discount**(idx+1)*r
            rewards_list.append(reward)
        return np.mean(rewards_list)

    def recover_q_sa(env, net, s, penalize_m=False):
        rows=len(s)
        dA=env.num_actions
        sclone=s.clone().detach().cpu().numpy()
        qvalues = np.zeros((rows, dA))
        for a in env.action_space:
            arows=np.repeat(a, rows)
            interm=np.zeros(rows)
            for m in env.mediator_space:
                #set P(m|s,a)
                mrows=np.repeat(m, rows)
                pm_sa=Pm_sa(sclone, arows, mrows)
                if penalize_m:
                    pm_sa=pm_sa-1.96*np.sqrt(getvar(sclone, arows)*pm_sa*(1-pm_sa))
                for atilde in env.action_space:
                    #set P(atilde|s) and W(s,atilde,m)
                    pa_s=Pa_s(sclone, np.repeat(atilde, rows))
                    interm+=net(s, torch.repeat_interleave(torch.tensor(atilde), rows).to(device), torch.repeat_interleave(torch.tensor(m), rows).to(device)).detach().cpu().numpy().squeeze()*pm_sa*pa_s
            qvalues[:,a]=interm  
        return qvalues
    
    ############################################FQI################################################
    def naivefqi(env, qnet, target_net, num_itrs, project_steps, batch_size, discount=0.99):
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(qnet.parameters())
        losses = []
        rewards = []
        rewards_list = []
        for i in range(num_itrs):
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            s_sample=torch.tensor(s[training_idx], dtype=torch.float).to(device)
            a_sample = torch.tensor(a[training_idx], dtype=torch.int64).to(device)
            sprime_sample = torch.tensor(sprime[training_idx], dtype=torch.float).to(device)
            r_sample=torch.tensor(r[training_idx], dtype=torch.float).to(device)

            q_values, index=torch.max(target_net(sprime_sample),dim=1)
            target = r_sample+discount*q_values
            
            pred = qnet(s_sample)
            pred = pred.gather(1, a_sample.reshape(-1,1)).squeeze()
            loss=mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Soft update of the target network's weights, Polyak averaging
            # θ' ← τ θ' + (1 −τ )θ
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = qnet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = tau*target_net_state_dict[key] + (1-tau)*policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
            
            if (i+1) % project_steps == 0:
                losses.append(float(loss.detach().cpu()))
                reward=trajectory_online_reward(env, target_net)
                rewards.append(reward)
                
                l=len(rewards)
                if l<=rwindow:
                    rewards_list.append(np.mean(rewards))
                else:
                    rewards_list.append(np.mean(rewards[-rwindow:]))
            
        return losses, rewards, rewards_list

    qnet=qNet(env).to(device)
    target_net = qNet(env).to(device)
    target_net.load_state_dict(qnet.state_dict())
    fqi_losses, fqi_rewards, fqi_rewards_cumulate = naivefqi(env, qnet, target_net, num_itrs, project_steps, batch_size)
    print("FQI online mean reward (std) with number of iterations %d and projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, np.mean(fqi_rewards), np.std(fqi_rewards)))

    
    ############################################CQL################################################
    def naivecql(env, qnet, target_net, num_itrs, project_steps, batch_size, cql_alpha=0.1, discount=0.99):
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(qnet.parameters())
        losses = []
        rewards = []
        rewards_list = []
        for i in range(num_itrs):
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            s_sample=torch.tensor(s[training_idx], dtype=torch.float).to(device)
            a_sample = torch.tensor(a[training_idx], dtype=torch.int64).to(device)
            sprime_sample = torch.tensor(sprime[training_idx], dtype=torch.float).to(device)
            r_sample=torch.tensor(r[training_idx], dtype=torch.float).to(device)

            q_values, index=torch.max(target_net(sprime_sample),dim=1)
            target = r_sample+discount*q_values
            
            pred = qnet(s_sample)
            logsumexp_qvalues = torch.logsumexp(pred, dim=-1)
            pred = pred.gather(1, a_sample.reshape(-1,1)).squeeze()
            cql_loss = logsumexp_qvalues - pred
            loss=mse_loss(pred, target)+cql_alpha*torch.mean(cql_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Soft update of the target network's weights, Polyak averaging
            # θ' ← τ θ' + (1 −τ )θ
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = qnet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = tau*target_net_state_dict[key] + (1-tau)*policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
            
            if (i+1) % project_steps == 0:
                losses.append(float(loss.detach().cpu()))
                reward=trajectory_online_reward(env, target_net)
                rewards.append(reward)
                
                l=len(rewards)
                if l<=rwindow:
                    rewards_list.append(np.mean(rewards))
                else:
                    rewards_list.append(np.mean(rewards[-rwindow:]))
            
        return losses, rewards, rewards_list

    qnet=qNet(env).to(device)
    target_net = qNet(env).to(device)
    target_net.load_state_dict(qnet.state_dict())
    cql_losses, cql_rewards, cql_rewards_cumulate = naivecql(env, qnet, target_net, num_itrs, project_steps, batch_size)
    print("CQL online mean reward (std) with number of iterations %d and projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, np.mean(cql_rewards), np.std(cql_rewards)))



    ###############################################################################################
    #######################################CAL & PESCAL############################################
    ###############################################################################################
    #Neural Net
    class mqNet(nn.Module):
        def __init__(self, env):
            super(mqNet, self).__init__()
            self.dim_state=env.dim_state
            self.fc_seqn = nn.Sequential(
                nn.Linear(self.dim_state+2,64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1))
        def forward(self, s, a_idx, m):
            x = torch.cat((s.reshape(-1,self.dim_state), a_idx.reshape(-1,1), m.reshape(-1,1)), dim=1)
            q = self.fc_seqn(x)
            return q
    
    def PESCAL(env, mqnet, target_net, num_itrs, project_steps, batch_size, penalize_m=False, discount=0.99):
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(mqnet.parameters())
        losses = []
        rewards = []
        rewards_list = []
        for i in range(num_itrs):  
            # start_time = time.time()
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size, replace=False)
            s_sample=torch.tensor(s[training_idx], dtype=torch.float32).to(device)
            a_sample = torch.tensor(a[training_idx], dtype=torch.int64).to(device)
            m_sample=torch.tensor(m[training_idx], dtype=torch.int64).to(device)
            sprime_sample = torch.tensor(sprime[training_idx], dtype=torch.float32).to(device)
            r_sample=torch.tensor(r[training_idx], dtype=torch.float32).to(device)
            
            qvalues=recover_q_sa(env, target_net, sprime_sample)
            maxqvalues = torch.tensor(np.max(qvalues, axis=1)).float().to(device)
            target = r_sample+discount*maxqvalues
            
            pred = mqnet(s_sample, a_sample, m_sample)
            loss = mse_loss(pred.squeeze(), target.float())
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Soft update of the target network's weights
            # θ' ← τ θ' + (1 −τ )θ
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = mqnet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = tau*target_net_state_dict[key] + (1-tau)*policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
            
            if (i+1) % project_steps == 0:
                losses.append(float(loss.detach().cpu()))
                reward=trajectory_online_reward(env, target_net, penalize_m, mediator=True)
                rewards.append(reward)
                l=len(rewards)
                if l<=rwindow:
                    rewards_list.append(np.mean(rewards))
                else:
                    rewards_list.append(np.mean(rewards[-rwindow:]))
               
        return losses, rewards, rewards_list
      
    #CAL
    mqnet=mqNet(env).to(device)
    target_net = mqNet(env).to(device)
    target_net.load_state_dict(mqnet.state_dict())
    cal_losses, cal_rewards, cal_rewards_cumulate = PESCAL(env, mqnet, target_net, num_itrs, project_steps, batch_size)
    print("FQI with mediator online mean reward (std) with number of iterations %d and projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, np.mean(cal_rewards), np.std(cal_rewards)))
    
    #PESCAL
    mqnet=mqNet(env).to(device)
    target_net = mqNet(env).to(device)
    target_net.load_state_dict(mqnet.state_dict())
    pescal_losses, pescal_rewards, pescal_rewards_cumulate = PESCAL(env, mqnet, target_net, num_itrs, project_steps, batch_size, penalize_m=True)
    print("pFQI with mediator online mean reward (std) with number of iterations %d and projection steps %d: \n%f+-(%f)" % (num_itrs, project_steps, np.mean(pescal_rewards), np.std(pescal_rewards)))


    
    dictionary = {
        "fqi_rewards": fqi_rewards,
        "fqi_rewards_cumulate": fqi_rewards_cumulate,
        "fqi_losses" :fqi_losses,
        "cql_rewards": cql_rewards,
        "cql_rewards_cumulate": cql_rewards_cumulate,
        "cql_losses": cql_losses,
        
        "cal_rewards": cal_rewards,
        "cal_rewards_cumulate": cal_rewards_cumulate,
        "cal_losses": cal_losses,
        "pescal_rewards": pescal_rewards,
        "pescal_rewards_cumulate": pescal_rewards_cumulate,
        "pescal_losses": pescal_losses,
        
    }

    with open("./data/pescal_real_noc_"+str(ratio)+"_"+str(iters)+".json", "w") as outfile:
        json.dump(dictionary, outfile)
        
    end_time = time.time()
    print('Algorithm takes: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int)
    parser.add_argument('--rwindow', type=int, default=50)
    parser.add_argument('--ratio', type=float)
    args = parser.parse_args()
    train(args.iters, args.rwindow, args.ratio)
    

