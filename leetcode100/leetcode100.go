package leetcode100

// https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&id=top-100-liked
// 1. 两数之和
func twoSum(nums []int, target int) []int {
	numMap := map[int]int{}
	for i, num := range nums {
		if j, ok := numMap[target-num]; ok {
			return []int{j, i}
		}
		numMap[num] = i
	}
	return nil
}
