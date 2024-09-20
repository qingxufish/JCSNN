import math
import copy

# 计算参搜索的所有组合
def combination(fullParameters):
    search_parameters = fullParameters.get('search_parameters')
    const_parameters = {}
    [const_parameters.update({key_name:fullParameters.get(key_name)}) for key_name in list(fullParameters.keys()) if key_name != 'search_parameters']

    keys = list(search_parameters.keys())
    result_array = [copy.deepcopy(const_parameters) for _ in search_parameters[keys[0]]]
    for item_index, item in enumerate(search_parameters[keys[0]]):
        result_array[item_index].update({keys[0]:item})

    for parameter_index in range(1,len(search_parameters)):
        pre_mutiple_lenth = len(result_array)
        result_array = [copy.deepcopy(item) for item in result_array * len(search_parameters[keys[parameter_index]])]
        for arr_index, _ in enumerate(result_array):
            value = search_parameters[keys[parameter_index]][math.floor(arr_index / pre_mutiple_lenth)]
            result_array[arr_index].update({keys[parameter_index]:value})

    return result_array

