import json
import os
import shutil
import random

from utils.generator import generator


class cenlight_pipeline:
    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            if self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])
    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)
    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._path_check()
        self._copy_conf_file()
        self._copy_anon_file()
        # test_duration
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)

    def run(self, multi_process=False):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()
        cenlight_runner = generator(self.dic_exp_conf, self.dic_agent_conf, self.dic_traffic_env_conf, self.dic_path)
        cenlight_runner.run()