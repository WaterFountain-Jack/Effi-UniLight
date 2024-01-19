# from ppo_agent import PPO
# from anon_env import AnonEnv
import csv

from env.env_Heterogeneous import AnonEnv
import os
import time
import numpy as np
from agent.Effi_UniLight import Effi_UniLight


class generator(object):
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.num_intersection = dic_traffic_env_conf["NUM_INTERSECTIONS"]
        self.model_name = self.dic_agent_conf["AGENT_NAME"]
        self.num_lanes = self.dic_traffic_env_conf["num_lanes"]
        self.num_phases = self.dic_traffic_env_conf["num_phases"]
        self.state_dim = self.compute_len_feature_irrgular() * self.num_intersection
        self.inter_need_choose_phase = []
        self.dic_traffic_env_conf["OLD_STATE_DIM"] = self.compute_len_old_state()
        self.old_dim = self.dic_traffic_env_conf["OLD_STATE_DIM"]

        print("\033[1;34m [---CONFIG---]  Model name: ", self.model_name, "\033[0m")
        print("\033[1;34m [---CONFIG---] State dim: ", self.state_dim, "\033[0m")
        print("\033[1;34m [---CONFIG---] Num of intersection: ", self.num_intersection, "\033[0m")
        print("\033[1;34m [---CONFIG---] State list : ", self.dic_traffic_env_conf["LIST_STATE_FEATURE"], "\033[0m")

        self.agent = Effi_UniLight(s_dim=self.state_dim, old_dim=self.old_dim, num_intersection=self.num_intersection,
                                   num_lanes=self.num_lanes, num_phases=self.num_phases)

    def compute_len_old_state(self):
        if 'ad_old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            return self.num_lanes * 3 + 1 + self.num_phases
        elif 'old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            return self.num_lanes * 2 + 1 + self.num_phases
    def compute_len_feature_irrgular(self):
        len_feature = 0
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature_name == "lane_num_vehicle":
                len_feature += self.num_lanes
            elif feature_name == "list_entering_num_l":
                len_feature += self.num_lanes
            elif "cur_phase" in feature_name:
                len_feature += self.num_lanes
            elif "time_this_phase" in feature_name:
                len_feature += 1
            elif "mask" in feature_name:
                len_feature += self.num_phases
            elif "traffic_movement_pressure_queue_efficient_lane_enter_running_part" in feature_name:
                len_feature += self.num_lanes * 2
            elif "all_phase_by_entering_lane" in feature_name:
                continue
            else:
                print("feature_name", feature_name, "no cont FEATURE_DIM")
        return len_feature
    def dict_to_list_irrgular(self, dict_state):
        dic_state_feature_arrays = {}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        for i in used_feature:
            dic_state_feature_arrays[i] = []
        for s in dict_state:
            for feature in used_feature:
                if feature == "cur_phase":
                    dic_state_feature_arrays[feature].append(s["all_phase_by_entering_lane"][s[feature][0]])
                else:
                    dic_state_feature_arrays[feature].append(s[feature])

        state_input0 = []
        for i in range(self.num_intersection):
            state_input0.append([])
            for feature in used_feature[0:4]:
                state_input0[i] = np.append(state_input0[i], dic_state_feature_arrays[feature][i])

        if 'ad_old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            state_input = [state_input0, np.array(dic_state_feature_arrays["all_phase_by_entering_lane"]),
                           np.array(dic_state_feature_arrays["mask"]),
                           np.array(dic_state_feature_arrays["ad_old_state"])]
        elif 'old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            state_input = [state_input0, np.array(dic_state_feature_arrays["all_phase_by_entering_lane"]),
                           np.array(dic_state_feature_arrays["mask"]), np.array(dic_state_feature_arrays["old_state"])]
        return state_input

    def run(self):
        for i in range(self.dic_exp_conf["NUM_ROUNDS"]):

            #test
            print("train: round %d starts" % i)
            path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_" + str(i))
            if not os.path.exists(path_to_log):
                os.makedirs(path_to_log)
            self.dic_traffic_env_conf["round"] = i
            env = AnonEnv(path_to_log=path_to_log,
                          path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                          dic_traffic_env_conf=self.dic_traffic_env_conf)
            [dict_state, self.inter_need_choose_phase] = env.reset()

            reset_env_start_time = time.time()
            done = False
            step_num = 0
            reset_env_time = time.time() - reset_env_start_time
            running_start_time = time.time()

            ##train
            while not done and step_num < int(
                    self.dic_exp_conf["RUN_COUNTS"] / self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
                print("step: ", step_num)
                step_start_time = time.time()
                list_state = self.dict_to_list_irrgular(dict_state)
                list_action = self.agent.choose_action(list_state, self.inter_need_choose_phase)
                print( " list action: ", list_action)
                dict_next_state, multi_reward, done, _ = env.step(list_action)
                print("time: {0}, running_time: {1}".format(env.get_current_time() - self.dic_traffic_env_conf["MIN_ACTION_TIME"],time.time() - step_start_time))

                self.agent.experience_storage(list_state[0], list_state[1], list_state[2], list_state[3],list_action, multi_reward)
                if (step_num % 40 == 0 and step_num != 0) or done == True:
                    list_state = self.dict_to_list_irrgular(dict_state)
                    self.agent.update(list_state[0],list_state[3])
                dict_state = dict_next_state
                step_num += 1
            self.agent.save(self.dic_path['PATH_TO_MODEL'] + "/model" + str(i) + ".pkl")
            running_time = time.time() - running_start_time
            log_start_time = time.time()
            print("start logging")
            env.bulk_log_multi_process()
            log_time = time.time() - log_start_time
            env.end_sumo()
            print("reset_env_time: ", reset_env_time)
            print("running_time: ", running_time)
            print("log_time: ", log_time)
