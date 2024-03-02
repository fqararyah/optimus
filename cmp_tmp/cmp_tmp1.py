def do_scheduling_dnnvm():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # batch size = 1
    batch_size.init(1)
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, d_fusion=True, womincost=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        res_parse(schedule_info_list, resource,
                  cost_model, sg, network,
                  loop_lower_bound,
                  './result/overall_experiment/dnnvm', arch_info, True)