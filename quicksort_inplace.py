def quicksort_inplace(arr, low=0, high=None):
    """快速排序 - 原地排序版本"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分区，返回基准元素的位置
        pivot_index = partition(arr, low, high)
        
        # 递归排序基准左右两边
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)
    
    return arr


def partition(arr, low, high):
    """分区函数"""
    pivot = arr[high]  # 选择最后一个元素作为基准
    i = low - 1  # 小于基准的元素的最后位置
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # 将基准元素放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# 测试
if __name__ == "__main__":
    test_list = [64, 34, 25, 12, 22, 11, 90, 5]
    print("原始数组:", test_list)
    sorted_list = quicksort_inplace(test_list)
    print("排序后:", sorted_list)
