import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))

import matplotlib.pyplot as plt

from fusion.scheduling import batch_size
from fusion.scheduling import Resource
from fusion.scheduling import LoopLowerBound
from fusion.scheduling import ScheduleGenerator
from fusion.scheduling import extract_info
from fusion.scheduling import CostModel
from fusion.scheduling import res_parse

from fusion.nn_models import import_network


arch_file = './fusion/arch/3_level_mem_512KB.json'
dataflow_file = './fusion/dataflow/fareed_df.json'

model_names = ['mobilenet', 'resnet50']


def do_scheduling_optimus(model_name, buffers):
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    num_bytes = arch_info["precision"] / 8

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # Network.
    batch_size.init(1)
    network = import_network(model_name)
    # [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
    access_list = []
    for bf in buffers:
        if type(arch_info["capacity"][1]) is list:
            arch_info["capacity"][1] = [bf * 1024 / num_bytes] * \
                len(arch_info["capacity"][1])
        else:
            arch_info["capacity"][1] = bf * 1024 / num_bytes
        resource = Resource.arch(arch_info)
        print("\n===========================================================")
        print('buffer size: {}(KB)'.format(bf))
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        #fareed
        for section_info in schedule_info_list:
            layer_group = section_info[0]
            print(layer_group)
            group_str = '['
            for layer_name in layer_group:
                if layer_name in network.layer_indeices_dict:
                    group_str += ' ' + str(network.layer_indeices_dict[layer_name]) + ','
            group_str = group_str[:-1] + ']'
            print(group_str)
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/buffer_size/hafs', arch_info, is_access=True, print_groups=True)
        access_list.append(access)
    x = [str(bf) for bf in buffers]
    access_list = [access / 1024 for access in access_list]
    return x, access_list


def do_scheduling_dnnvm(model_name, buffers):
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    num_bytes = arch_info["precision"] / 8

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # Network.
    batch_size.init(1)
    network = import_network(model_name)
    #[16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
    access_list = []
    for bf in buffers:
        if type(arch_info["capacity"][1]) is list:
            arch_info["capacity"][1] = [bf * 1024 / num_bytes] * \
                len(arch_info["capacity"][1])
        else:
            arch_info["capacity"][1] = bf * 1024 / num_bytes
        resource = Resource.arch(arch_info)
        print("\n===========================================================")
        print('buffer size: {}(KB)'.format(bf))
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model,
                               loop_lower_bound, d_fusion=True, womincost=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/buffer_size/dnnvm', arch_info, True, is_access=True)
        access_list.append(access)
    x = [str(bf) for bf in buffers]
    access_list = [access / 1024 for access in access_list]
    return x, access_list


def do_scheduling_efficients(model_name, buffers):
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    num_bytes = arch_info["precision"] / 8

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # Network.
    batch_size.init(1)
    network = import_network(model_name)
    #[16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
    access_list = []
    for bf in buffers:
        if type(arch_info["capacity"][1]) is list:
            arch_info["capacity"][1] = [bf * 1024 / num_bytes] * \
                len(arch_info["capacity"][1])
        else:
            arch_info["capacity"][1] = bf * 1024 / num_bytes
        resource = Resource.arch(arch_info)
        print("\n===========================================================")
        print('buffer size: {}(KB)'.format(bf))
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model,
                               loop_lower_bound, z_fusion=True, womincost=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/buffer_size/efficients', arch_info, True, is_access=True)
        access_list.append(access)
    x = [str(bf) for bf in buffers]
    access_list = [access / 1024 for access in access_list]
    return x, access_list


def do_scheduling_wofusion(model_name, buffers):
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    num_bytes = arch_info["precision"] / 8

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # Network.
    batch_size.init(1)
    network = import_network(model_name)
    
    #[16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
    access_list = []
    for bf in buffers:
        if type(arch_info["capacity"][1]) is list:
            arch_info["capacity"][1] = [bf * 1024 / num_bytes] * \
                len(arch_info["capacity"][1])
        else:
            arch_info["capacity"][1] = bf * 1024 / num_bytes
        resource = Resource.arch(arch_info)
        print("\n===========================================================")
        print('buffer size: {}(KB)'.format(bf))
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model,
                               loop_lower_bound, wofusion=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/buffer_size/woFusion', arch_info, True, is_access=True)
        access_list.append(access)
    x = [str(bf) for bf in buffers]
    access_list = [access / 1024 for access in access_list]
    return x, access_list


def main():
    """
    Main function.
    """
    buffers = [256, 384, 512, 768, 1024]
    for model_name in model_names:
        plt.figure(figsize=(10, 4))
        print('\n')
        print("*" * 60)
        print('HaFS:')
        x, access_list = do_scheduling_optimus(model_name, buffers)
        plt.plot(x, access_list, label="HaFS")
        print('\n')
        print("*" * 60)
        print('Efficient-S:')
        _, access_list = do_scheduling_efficients(model_name, buffers)
        plt.plot(x, access_list, label="Efficient-S")
        print('\n')
        print("*" * 60)
        print('DNNVM:')
        _, access_list = do_scheduling_dnnvm(model_name, buffers)
        plt.plot(x, access_list, label="DNNVM")
        print('\n')
        print("*" * 60)
        print('w/o Fusion:')
        _, access_list = do_scheduling_wofusion(model_name, buffers)
        plt.plot(x, access_list, label="w/o Fusion")

        plt.ylabel("DRAM access volume(GB)")
        plt.xlabel("on-chip buffer size(KB)")
        plt.ylim(0.0, 0.12)
        plt.legend()
        plt.savefig('./result/buffer_size/{}buffer_size.png'.format(model_name))
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())