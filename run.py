import config.config as config
import copy
from utils.pipeline import cenlight_pipeline
from multiprocessing import Process
import time
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    name = 'Effi_UniLight'

    parser.add_argument("--memo", type=str, default=name )
    parser.add_argument("--road_net", type=str, default='')
    #'jinan' 'hangzhou' 'newyork16_3 'newyork28_7' 'manhattan' 'SH1' 'SH2'
    parser.add_argument("--data", type=str, default='jinan')
    parser.add_argument("-workers",    type=int, default=3)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("--mod", type=str, default=name)
    parser.add_argument("--cnt",type=int, default=3600)
    parser.add_argument("--num_rounds",type=int, default=100)
    return parser.parse_args()



def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = cenlight_pipeline(dic_exp_conf=dic_exp_conf, # experiment config
                   dic_agent_conf=dic_agent_conf, # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf, # the simolation configuration
                   dic_path=dic_path # where should I save the logs?
                   )
    ppl.run(multi_process=False)

    print("pipeline_wrapper end")
    return

def main(in_args=None):
    num_lanes = 0
    num_phases = 0
    if "jinan" in in_args.data:
        num_lanes = 12
        num_phases = 9
        traffic_file_list = ["anon_3_4_jinan_real.json"]
        num_intersections = 12
        template="Jinan"
        in_args.road_net='3_4'
    elif "hangzhou" in in_args.data:
        num_lanes = 12
        num_phases = 9
        traffic_file_list = ["anon_4_4_hangzhou_real.json"]
        num_intersections = 16
        template = "Hangzhou"
        in_args.road_net='4_4'
    elif "newyork16_3" in in_args.data:
        num_lanes = 12
        num_phases = 9
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        num_intersections = 16*3
        template = "NewYork"
        in_args.road_net = '16_3'
    elif "newyork28_7" in in_args.data:
        num_lanes = 12
        num_phases = 9
        traffic_file_list = ["anon_28_7_newyork_real.json"]
        num_intersections = 28*7
        template = "NewYork"
        in_args.road_net = '28_7'
    elif "manhattan" in in_args.data:
        num_lanes = 42
        num_phases = 10
        num_intersections = 3763
        traffic_file_list = ["manhattan_21430.json"]
        template="manhattan"
        in_args.road_net = 'manhattan'
    elif "SH1" in in_args.data:
        num_lanes = 22
        num_phases = 9
        num_intersections = 6
        traffic_file_list = ["flow.json",]
        template = "SH1"
        in_args.road_net = 'SH1'
    elif "SH2" in in_args.data:
        num_lanes = 16
        num_phases = 9
        num_intersections = 8
        traffic_file_list = ["flow.json",]
        template = "SH2"
        in_args.road_net = 'SH2'



    process_list = []

    for traffic_file in traffic_file_list:
        dic_exp_conf_extra = {
            "RUN_COUNTS": in_args.cnt,
            "MODEL_NAME": in_args.mod,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic
            "ROADNET_FILE": "roadnet_{0}.json".format(in_args.road_net),
            "NUM_ROUNDS": in_args.num_rounds,
        }

        dic_traffic_env_conf_extra = {
            "OBS_LENGTH": 111,
            "OLD_STATE_LENGTH":10,
            "OLD_STATE_DIM":0,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "MODEL_NAME": in_args.mod,
            "num_lanes": num_lanes,
            "num_phases": num_phases,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(in_args.road_net),
            "DIC_REWARD_INFO": {
                "flickering": 0,#-5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,#-1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0,
                "time_punish":0
            },
        }

        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template,  str(in_args.road_net)),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        dic_agent_conf = getattr(config, "DIC_{0}_AGENT_CONF".format(in_args.mod.upper()))
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        deploy_dic_traffic_env_conf['LIST_STATE_FEATURE'] = dic_agent_conf['LIST_STATE_FEATURE']

        if in_args.multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                            dic_agent_conf=dic_agent_conf,
                            dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                            dic_path=deploy_dic_path)
    if in_args.multi_process:
            for i in range(0, len(process_list), in_args.workers):
                i_max = min(len(process_list), i + in_args.workers)
                for j in range(i, i_max):
                    print(j)
                    print("start_traffic")
                    process_list[j].start()
                    print("after_traffic")
                for k in range(i, i_max):
                    print("traffic to join", k)
                    process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo

if __name__ == "__main__":
    args = parse_args()
    main(args)



