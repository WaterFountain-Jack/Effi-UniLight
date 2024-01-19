import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
from multiprocessing import Process


class Intersection:
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, lanes_length_dict
                 ,num_phases,num_lanes):

        self.inter_name = inter_id
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.lane_length = lanes_length_dict
        self.obs_length = dic_traffic_env_conf["OBS_LENGTH"]

        self.all_phase_by_entering_lane = light_id_dict["all_phase_by_entering_lane"]
        self.all_phase_by_exiting_lane = light_id_dict["all_phase_by_exiting_lane"]
        self.list_exiting_lanes_of_entering_lanes = light_id_dict["list_exiting_lanes_of_entering_lanes"]

        self.dic_entering_idx_to_lanes  = light_id_dict["dic_entering_idx_to_lanes"]
        self.dic_exiting_idx_to_lanes  = light_id_dict["dic_exiting_idx_to_lanes"]
        self.num_phases = num_phases
        self.num_lanes = num_lanes

        if 'ad_old_state' in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            self.ad_old_state_length = dic_traffic_env_conf["OLD_STATE_LENGTH"]
            self.ad_old_state_dim = dic_traffic_env_conf["OLD_STATE_DIM"]
            self.ad_old_state_one = [0 for _ in range(self.ad_old_state_dim)]
            self.ad_old_state = [[0 for _ in range(self.ad_old_state_dim)] for _ in range(self.ad_old_state_length)]
        elif 'old_state' in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            self.old_state_length = dic_traffic_env_conf["OLD_STATE_LENGTH"]
            self.old_state_dim = dic_traffic_env_conf["OLD_STATE_DIM"]
            self.old_state_one = [0 for _ in range(self.old_state_dim )]
            self.old_state = [[0 for _ in range(self.old_state_dim)] for _ in range(self.old_state_length)]


        self.phases_lenth = len(self.all_phase_by_entering_lane)
        self.mask = [True for i in range(self.phases_lenth)] + [False for i in range(self.num_phases - self.phases_lenth)]
        self.mask = self.mask[0:self.num_phases]



        self.list_entering_lanes = light_id_dict["list_entering_lanes"]
        self.list_exiting_lanes = light_id_dict["list_exiting_lanes"]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]


        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode="a", header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time, path_to_log):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = action
            if self.current_phase_index == self.next_phase_to_set_index:
                pass
            else:
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state["get_lane_vehicles"][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step_in)
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step_in) - set(self.list_lane_vehicle_previous_step_in))
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(set(self.list_lane_vehicle_current_step_in) -
                                         set(self.list_lane_vehicle_previous_step_in))
        else:
            list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step_in) -
                                         set(self.list_lane_vehicle_current_step_in))
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def to_num_lanes(self,x):
        x += [0 for i in range(self.num_lanes - len(x))]
        return x
    def to_all_phases(self,x):
        # complementary 0 operation
        for i in range(len(x)):
            if len(x[i]) < self.num_lanes:
                x[i] += [0 for i in range(self.num_lanes - len(x[i]))]
        for i in range(self.num_phases - len(x)):
            x.append([0 for i in range(self.num_lanes)])
        return x[0:self.num_phases]
    def _get_part_observations(self,lane_vehicles, vehicle_distance, vehicle_speed,
                               lane_length, obs_length, list_lanes):
        """
            Input: lane_vehicles :      Dict{lane_id    :   [vehicle_ids]}
                   vehicle_distance:    Dict{vehicle_id :   float(dist)}
                   vehicle_speed:       Dict{vehicle_id :   float(speed)}
                   lane_length  :       Dict{lane_id    :   float(length)}
                   obs_length   :       The part observation length
                   list_lanes   :       List[lane_ids at the intersection]
        :return:
                    part_vehicles:      Dict{ lane_id, [vehicle_ids]}
        """
        # get vehicle_ids and speeds
        first_part_num_vehicle = {}
        first_part_queue_vehicle = {}  # useless, at the begin of lane, there is no waiting vechiles
        last_part_num_vehicle = {}
        last_part_queue_vehicle = {}

        for lane in list_lanes:
            first_part_num_vehicle[lane] = []
            first_part_queue_vehicle[lane] = []
            last_part_num_vehicle[lane] = []
            last_part_queue_vehicle[lane] = []
            last_part_obs_length = lane_length[lane] - obs_length
            for vehicle in lane_vehicles[lane]:
                """ get the first part of obs
                    That is vehicle_distance <= obs_length 
                """
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        first_part_queue_vehicle[lane].append(vehicle)

                """ get the last part of obs
                    That is  lane_length-obs_length <= vehicle_distance <= lane_length 
                """
                if temp_v_distance >= last_part_obs_length:
                    last_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle
    def _get_part_traffic_movement_features(self):
        """
        return: part_traffic_movement_pressure_num:     both the end and the beginning of the lane
                part_patrric_movement_pressure_queue:   all at the end of the road
                part_entering_running_vehicles:         part obs of the running vehicles
        """
        f_p_num, l_p_num, l_p_q = self._get_part_observations(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                              vehicle_distance=self.dic_vehicle_distance_current_step,
                                                              vehicle_speed=self.dic_vehicle_speed_current_step,
                                                              lane_length=self.lane_length,
                                                              obs_length=self.obs_length,
                                                              list_lanes=self.list_lanes)
        """calculate traffic_movement_pressure with part queue"""
        list_entering_part_queue = [len(l_p_q[lane]) for lane in self.list_entering_lanes]

        """calculate traffic_movement_pressure with part num vehicle"""
        list_entering_num_l = [len(l_p_num[lane]) for lane in self.list_entering_lanes]
        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)
        return part_entering_running, list_entering_part_queue,list_entering_num_l

    def _update_feature(self):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["lane_num_vehicle"] = self.to_num_lanes(self._get_lane_num_vehicle(self.list_entering_lanes))
        dic_feature["all_phase_by_entering_lane"] = self.to_all_phases(self.all_phase_by_entering_lane)
        dic_feature["mask"] = self.mask
        dic_feature["pressure"] = self._get_pressure()
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        enter_running_part, lepq, list_entering_num_l = self._get_part_traffic_movement_features()

        lane_enter_running_part = self.to_num_lanes(list(enter_running_part))
        dic_feature["lane_num_waiting_vehicle_in"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_waiting_vehicle_out"] = self._get_lane_queue_length(self.list_exiting_lanes)

        traffic_movement_pressure_queue_efficient = self.to_num_lanes(self._get_traffic_movement_pressure_efficient(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"]))

        dic_feature["traffic_movement_pressure_queue_efficient_lane_enter_running_part"] =  traffic_movement_pressure_queue_efficient + lane_enter_running_part

        if 'ad_old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            self.ad_old_state_one = dic_feature["all_phase_by_entering_lane"][dic_feature["cur_phase"][0]] + dic_feature[
                "time_this_phase"] + dic_feature["traffic_movement_pressure_queue_efficient_lane_enter_running_part"] + \
                                 dic_feature["mask"][0:self.num_phases]

            self.ad_old_state = self.ad_old_state[1:self.ad_old_state_length]
            self.ad_old_state.append(self.ad_old_state_one)
            dic_feature["ad_old_state"] = self.ad_old_state
        elif 'old_state' in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            self.old_state_one = dic_feature["all_phase_by_entering_lane"][dic_feature["cur_phase"][0]] + dic_feature[
                "time_this_phase"] + dic_feature["lane_num_vehicle"] + dic_feature["mask"][0:self.num_phases]

            self.old_state = self.old_state[1:self.old_state_length]
            self.old_state.append(self.old_state_one)
            dic_feature["old_state"] = self.old_state
        self.dic_feature = dic_feature

    def _get_traffic_movement_pressure_efficient(self,enterings, exitings):
        tmp = []

        exitings_np = np.array(exitings)
        np_list_exiting_lanes_of_entering_lanes = self.list_exiting_lanes_of_entering_lanes
        for i in range (len(enterings)):
            if sum(np_list_exiting_lanes_of_entering_lanes[i]) == 0:
                tmp = tmp + [0]
            else:
                tmp = tmp + [enterings[i] - sum(np_list_exiting_lanes_of_entering_lanes[i] * exitings_np) / sum(np_list_exiting_lanes_of_entering_lanes[i]) ]
        return tmp

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]



    # def _get_pressure(self, l_v_in, l_v_out):
    #     return list(np.array(l_v_in) - np.array(l_v_out))
    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for
                     state_feature_name in list_state_features}
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row


    def get_reward(self, dic_reward_info):
        # customize your own reward
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"])) # np.sum(self.dic_feature["pressure"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class AnonEnv:

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.road_node_dic = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None
        self.num_lanes = self.dic_traffic_env_conf["num_lanes"]
        self.num_phases = self.dic_traffic_env_conf["num_phases"]


        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            """ include the yellow time in action time """
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile":"roadnetLogFile"+str(self.dic_traffic_env_conf["round"])+".json",
            "replayLogFile": "replayLogFile"+str(self.dic_traffic_env_conf["round"])+".txt",
        }

        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)


        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=4)

        [self.traffic_light_node_dict,self.inter_need_choose_phase] = self._adjacency_extraction()

        _, self.lane_length = self.get_lane_length()


        self.list_intersection = [Intersection(i, self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict[i],
                                               self.path_to_log,
                                               self.lane_length,
                                               self.num_phases,
                                               self.num_lanes)
                                  for i in self.traffic_light_node_dict.keys()]


        self.list_inter_log = [[] for _ in range(len(self.traffic_light_node_dict.keys()))]
        self.id_to_index = {}

        count = 0
        for i in self.traffic_light_node_dict.keys():
            self.id_to_index[i] = count
            count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

        # for inter in self.list_intersection:
        #     inter.update_current_measurements_map(self.system_states)
        state, done = self.get_state()
        return [state,self.inter_need_choose_phase]

    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()


            if i == 0:
                print("time: {0}".format(instant_time))

            self._inner_step(action_in_sec)


            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)

        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):

        for inter in self.list_intersection:
            inter.update_previous_measurements()

        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )


        # run one step

        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):

            self.eng.next_step()

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = False
        return list_state, done

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        road_node_dic = {}
        inter_need_choose_phase = []
        self.total_lane_num = 0

        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])



        with open('{0}'.format(file)) as json_data:

            net = json.load(json_data)

            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,

                                                            "roads":inter['roads'],
                                                            "roadLinks": inter["roadLinks"],
                                                            "lightphases": inter["trafficLight"]["lightphases"],
                                                            "entering_roads":[],
                                                            "exiting_roads":[],
                                                            "list_entering_lanes":[],
                                                            "list_exiting_lanes":[],

                                                            # 记录每一入lane对应出lane的idx
                                                            "list_exiting_lanes_of_entering_lanes": [],
                                                            "dic_entering_lanes_to_idx":{},
                                                            "dic_exiting_lanes_to_idx":{},
                                                            "dic_entering_idx_to_lanes": {},
                                                            "dic_exiting_idx_to_lanes": {},
                                                            "all_phase_by_entering_lane":[],
                                                            "all_phase_by_exiting_lane":[],

                                                            }

                    set_entering_roads = set()
                    set_exiting_roads = set()
                    # Used to hold temporary roadlinks, i.e., a single traffic moviment for each intersection.
                    list_roadLinks = inter["roadLinks"]
                    # Used to record incoming and outgoing paths in this node
                    for roadLink in list_roadLinks:
                        set_entering_roads.add(roadLink["startRoad"])
                        set_exiting_roads.add(roadLink["endRoad"])
                    for ro in traffic_light_node_dict[inter['id']]["roads"]:
                        if ro in set_entering_roads:
                            traffic_light_node_dict[inter['id']]["entering_roads"].append(ro)
                        elif ro in set_exiting_roads:
                            traffic_light_node_dict[inter['id']]["exiting_roads"].append(ro)



            # Collecting information on roads

            for road in net['roads']:
                if road['id'] not in road_node_dic.keys():
                    road_node_dic[road['id']] = {}

                # Record a road from which intersection to which intersection.
                road_node_dic[road['id']]['from'] = road['startIntersection']
                road_node_dic[road['id']]['to'] = road['endIntersection']

                # Record how many lanes there are on a road.
                road_node_dic[road['id']]['lan_num'] = len(road['lanes'])
                self.total_lane_num += len(road['lanes'])
                



            inter_id_to_index = {}

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

        for key,value in traffic_light_node_dict.items():

            idx = 0
            for entering_road in value['entering_roads']:
                for i in range (road_node_dic[entering_road]['lan_num']):
                    traffic_light_node_dict[key]["list_entering_lanes"].append("{0}_{1}".format(entering_road,i))
                    traffic_light_node_dict[key]["dic_entering_idx_to_lanes"][idx] = "{0}_{1}".format(entering_road,i)
                    traffic_light_node_dict[key]["dic_entering_lanes_to_idx"]["{0}_{1}".format(entering_road, i)] = idx
                    idx = idx + 1
            idx = 0
            for exiting_road in value['exiting_roads']:
                for i in range (road_node_dic[exiting_road]['lan_num']):
                    traffic_light_node_dict[key]["list_exiting_lanes"].append("{0}_{1}".format(exiting_road,i))
                    traffic_light_node_dict[key]["dic_exiting_idx_to_lanes"][idx] = "{0}_{1}".format(exiting_road,i)
                    traffic_light_node_dict[key]["dic_exiting_lanes_to_idx"]["{0}_{1}".format(exiting_road, i)] = idx
                    idx = idx + 1



            traffic_light_node_dict[key]["list_exiting_lanes_of_entering_lanes"] = [set() for _ in range(len(traffic_light_node_dict[key]["list_entering_lanes"]))]


            for lightphase in value["lightphases"]:

                a_phase_by_entering_lane = [0 for i in range(len(value["list_entering_lanes"]))]
                a_phase_by_exiting_lane = [0 for i in range(len(value["list_exiting_lanes"]))]


                for availableRoadLink in lightphase["availableRoadLinks"]:

                    link = value["roadLinks"][availableRoadLink]
                    startRoad = link["startRoad"]
                    endRoad = link["endRoad"]
                    for laneLink in link["laneLinks"]:

                        startLaneIndex = laneLink["startLaneIndex"]
                        endLaneIndex = laneLink["endLaneIndex"]
                        entering_lane = "{0}_{1}".format(startRoad,startLaneIndex)
                        exiting_lane = "{0}_{1}".format(endRoad,endLaneIndex)
                        if(link["type"] != "turn_right"):
                            a_phase_by_entering_lane[value["dic_entering_lanes_to_idx"][entering_lane]] = 1
                            a_phase_by_exiting_lane[value["dic_exiting_lanes_to_idx"][exiting_lane]] = 1
                        traffic_light_node_dict[key]["list_exiting_lanes_of_entering_lanes"][value["dic_entering_lanes_to_idx"][entering_lane]].add(value["dic_exiting_lanes_to_idx"][exiting_lane])


                traffic_light_node_dict[key]["all_phase_by_entering_lane"].append(a_phase_by_entering_lane)
                traffic_light_node_dict[key]["all_phase_by_exiting_lane"].append(a_phase_by_exiting_lane)

            for idx_list_exiting_lanes_of_entering_lanes in range(len(traffic_light_node_dict[key]["list_exiting_lanes_of_entering_lanes"])):
                a_list_exiting_lanes_of_entering_lanes = [0 for _ in range(len(value["list_exiting_lanes"]))]
                for set_value in traffic_light_node_dict[key]["list_exiting_lanes_of_entering_lanes"][idx_list_exiting_lanes_of_entering_lanes]:
                    a_list_exiting_lanes_of_entering_lanes[set_value] = 1
                traffic_light_node_dict[key]["list_exiting_lanes_of_entering_lanes"][idx_list_exiting_lanes_of_entering_lanes] = a_list_exiting_lanes_of_entering_lanes




            ##free up memory
            traffic_light_node_dict[key]["lightphases"] = {}
            if (len(value["all_phase_by_entering_lane"]) > 1):
                inter_need_choose_phase = inter_need_choose_phase + [True]
            else:
                inter_need_choose_phase = inter_need_choose_phase + [False]



        return [traffic_light_node_dict,inter_need_choose_phase]



    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for i in range(len(road["lanes"])):
                lane_id = road['id'] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict
    def end_sumo(self):
        print("anon process end")
        pass
