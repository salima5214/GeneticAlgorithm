import os
import time
import yaml
from csv import writer

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
# with open('record_{}.csv'.format(time.strftime("%Y-%m-%d", time.localtime())), 'a', newline='') as f_object:  
#     writer_object = writer(f_object)
#     record = ["time :{}, task:{}, mask_order:{}, bound_min:{}, bound_max:{}, init_mean:{}, fitess_top_proportion:{}, np_seed:{}, cma_seed:{}, RM_time:{}".format(
#         time.strftime("%m%d-%H%M", time.localtime()),
#         config['op']['mask_order'],
#         config['hp']['dim'],
#         config['hp']['bound_min'],
#         config['hp']['bound_max'],
#         config['hp']['init_mean'],
#         config['hp']['fitess_top_proportion'],
#         config['hp']['np_seed'],
#         config['hp']['cma_seed'],
#         config['hp']['RM_time'])]
#     writer_object.writerow(record)  
#     f_object.close()



def write_info():
    with open('record_{}.csv'.format(time.strftime("%Y-%m-%d", time.localtime())), 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        record = ["time :{}, task:{}, dimension:{}, components:{}, mask_order:{}, init_mean:{}, fitess_top_proportion:{}, np_seed:{}, cma_seed:{}, RM_time:{}".format(
            time.strftime("%m%d-%H%M", time.localtime()),
            config['op']['mask_order'],
            dim, # ! dim
            config['hp']['components'],
            config['op']['mask_order'],
            config['hp']['init_mean'],
            config['hp']['fitess_top_proportion'],
            config['hp']['np_seed'],
            config['hp']['cma_seed'],
            RM_time)] # ! RM_time
        writer_object.writerow(record)  
        f_object.close()

dim = 8
while dim <= 8:
    for RM_time in range(0, 1):
        write_info()
        for i in range(1): 
            os.system("python3 ./system_cmaes.py {} {}".format(dim, RM_time))
    dim = dim + 1