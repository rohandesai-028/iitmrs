from dqn_tf2 import Agent
#from sac import SAC_Agent
import numpy as np # type: ignore
import gym # type: ignore
#from utils import plotLearning
import tensorflow as tf # type: ignore
from scheduler_mac import Scheduler
import matplotlib.pyplot as plt # type: ignore
import h5py # type: ignore
    
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    
    # CONSTANTS 
    TOTAL_EMBB_USERS = 10
    lr = 0.0001
    n_games = 2500
    cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
    cqi_embb_u = np.clip(cqi_embb_user, 0, 28)
    sch1 = Scheduler(cqi_embb_user=cqi_embb_u, embb_no=TOTAL_EMBB_USERS)
    sch1.update_cqi_br()
    scores = []
    eps_history = []
    avg_array = []
    y = []
    total_latency = 0
    total_served = 0
    
    #GUI INPUTS
    print("\n---------------Welcome to the Scheduler---------------------")
    print("\nPick simulation mode: \n0 for single pass(1 urllc arrival rate),\n1 for continuous mode(Arrival rate from 0 to 100 in steps of 5)")
    sim_mode = int(input())
    print("Pick the choice of algorithm: \n0 for Puncturing(No DQN), \n1 for DQN")
    algorithm_choice = int(input())
    print("Train the model or load the model? \n0 for train, \n1 for load")
    train_or_load = int(input())
    
    # MODEL TRAINING/LOAD AGENT INSTANTIATION
    if train_or_load == 0: #TRAIN MODE
        print("enter save file name: ")
        savefile_name = input() + ".h5"
        agent = Agent(gamma=0.3, epsilon=1, lr=lr, 
            input_dims=sch1.observation_space.shape,
            n_actions=sch1.action_space.n, mem_size=40000, batch_size=32,epsilon_dec=0.0003,
            epsilon_end=0.01,
            fname=savefile_name)
    else: #LOAD AND TEST MODEL MODE
        agent = Agent(gamma=0.3, 
                      epsilon=0, 
                      lr=lr, 
                      input_dims=sch1.observation_space.shape,
                      n_actions=sch1.action_space.n, 
                      mem_size=40000, 
                      batch_size=32,
                      epsilon_dec=0,
                      epsilon_end=0)
        agent.load_model()
        n_games = 200
    
    #AGENT TRAINING OVER 1 URRLC ARRIVAL RATE
    if sim_mode == 0:
        print("Pick an Arrival Rate in following range: 0:5:100")
        print("Enter index of step in Arrival rate")
        ar_index = int(input())
        for i in range(n_games):
            done = False
            score = 0
            observation = sch1.reset()
            env_ctr = 1
            action_counter = 0
            score_history_per_episode = []
            mean_score = 0
    ######################################################################
    #      Start of N actions in each episode
    ######################################################################        
            while not done:
                if env_ctr % 3 == 0:
                    sch1.update_cqi_br()
                env_ctr += 1
                action_counter += 1
                action = agent.choose_action(observation)

                if(algorithm_choice == 0): # for Puncturing mode of operation with no DQN allocation
                    action = 0
                observation_, reward, done = sch1.step(action, ar_index)
                score += reward
                score_history_per_episode.append(score)
                mean_score = sum(score_history_per_episode)/len(score_history_per_episode)

                print("Action Counter = ",action_counter,"Action =",action,"Arrivals =",sch1.urllc_ct,
                    "E_DR =%.2f"%sum(sch1.inst_data),
                    "Queue_length =",sch1.queue_len(),
                    "score =",score, "mean score = ",mean_score)
                agent.store_transition(observation, action, reward, observation_, done)
                observation = observation_
                agent.learn()
    ######################################################################
    #      End of N actions in each episode
    ######################################################################        
            
            agent.episode_counter_in(i)
            eps_history.append(agent.epsilon)
            scores.append(score)
            
            
            avg_score = np.mean(scores[-20:])
            #avg_array.append(moving_avg(score))
            avg_array.append(mean_score)
            print('episode: ', i, 'score %.2f' % score,
                    'average_score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon,
                    'epsilon_length = ', len(eps_history),
                    '\nepsilon_history =',eps_history)
            print("\n\n")
            avg_latency, served = sch1.get_urllc_latency_stats()
            total_latency += avg_latency * served
            total_served += served
            y.append(sch1.urllc_served_per_slot)
        #filename = 'lunarlander_tf2.png'
        x = [i+1 for i in range(n_games)]
        xaxis = [j+1 for j in range(len(avg_array))]
        agent.save_model()
        overall_avg_latency = total_latency / total_served
        
    else:
        ar_index = 0
        latency_stats = []
        urllc_dr_arr = []
        embb_dr_arr = []
        for ar_index in range(0, 19, 1):
            for i in range(n_games):
                done = False
                score = 0
                observation = sch1.reset()
                env_ctr = 1
                action_counter = 0
                score_history_per_episode = []
                mean_score = 0
        ######################################################################
        #      Start of N actions in each episode
        ######################################################################        
                while not done:
                    if env_ctr % 3 == 0:
                        sch1.update_cqi_br()
                    env_ctr += 1
                    action_counter += 1
                    action = agent.choose_action(observation)

                    if(algorithm_choice == 0): # for Puncturing mode of operation with no DQN allocation
                        action = 0
                    observation_, reward, done = sch1.step(action, ar_index)
                    score += reward
                    score_history_per_episode.append(score)
                    mean_score = sum(score_history_per_episode)/len(score_history_per_episode)

                    print("Action Counter = ",action_counter,"Action =",action,"Arrivals =",sch1.urllc_ct,
                        "E_DR =%.2f"%sum(sch1.inst_data),
                        "Queue_length =",sch1.queue_len(),
                        "score =",score, "mean score = ",mean_score)
                    agent.store_transition(observation, action, reward, observation_, done)
                    observation = observation_
                    agent.learn()
        ######################################################################
        #      End of N actions in each episode
        ######################################################################        
                
                agent.episode_counter_in(i)
                eps_history.append(agent.epsilon)
                scores.append(score)
                
                
                avg_score = np.mean(scores[-20:])
                #avg_array.append(moving_avg(score))
                avg_array.append(mean_score)
                print('episode: ', i, 'score %.2f' % score,
                        'average_score %.2f' % avg_score,
                        'epsilon %.2f' % agent.epsilon,
                        'epsilon_length = ', len(eps_history),
                        '\nepsilon_history =',eps_history)
                print("\n\n")
                avg_latency, served = sch1.get_urllc_latency_stats()
                total_latency += avg_latency * served
                total_served += served
                y.append(sch1.urllc_served_per_slot)
            #filename = 'lunarlander_tf2.png'
            x = [i+1 for i in range(n_games)]
            xaxis = [j+1 for j in range(len(avg_array))]
            #agent.save_model()
            overall_avg_latency = total_latency / total_served
            print("Total URLLC served: ", total_served
                , "Total URLLC Latency: ", total_latency)
            print(f"\nAverage achieved URLLC latency over entire test period: {overall_avg_latency:.2f} slots")
            print("Total EMBB DR: \t",sch1.total_embb_dr)
            print("Total URLLC DR: \t",sch1.urllc_data_rate)
            print("urllc served per slot \t",sum(y)/len(y))
            # plt.plot(xaxis,np.convolve(avg_array, np.ones(50)/float(50), 'same'))
            # plt.xlabel('Episode')
            # plt.ylabel('Reward')
            # plt.show()
            latency_stats.append(overall_avg_latency)
            urllc_dr_arr.append(sch1.urllc_data_rate)
            embb_dr_arr.append(sch1.total_embb_dr)
        print("AVG LATENCY in every Arrival Rate",latency_stats)
        print("AVG URLLC DR in every Arrival Rate",urllc_dr_arr)
        print("AVG EMBB DR in every Arrival Rate",embb_dr_arr)
    
