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

// https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&id=top-100-liked
// 49. 字母异位词分组
func groupAnagrams(strs []string) [][]string {
	strMap := map[[26]int][]string{}
	for _, str := range strs {
		var strArr [26]int
		for _, s := range str {
			strArr[s-'a']++
		}
		strMap[strArr] = append(strMap[strArr], str)
	}
	var res [][]string
	for _, v := range strMap {
		res = append(res, v)
	}
	return res
}

// https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&id=top-100-liked
// 128. 最长连续序列
func longestConsecutive(nums []int) int {
	numMap := map[int]bool{}
	for _, num := range nums {
		numMap[num] = true
	}
	var max int
	for _, num := range nums {
		if !numMap[num-1] {
			var cur int
			for numMap[num] {
				cur++
				num++
			}
			if cur > max {
				max = cur
			}
		}
	}
	return max
}
