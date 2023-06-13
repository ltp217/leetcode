package leetcode100

import "sort"

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

// https://leetcode.cn/problems/move-zeroes/?envType=study-plan-v2&id=top-100-liked
// 283. 移动零
func moveZeroes(nums []int) {
	var j int
	for i, num := range nums {
		if num != 0 {
			nums[i], nums[j] = nums[j], nums[i]
			j++
		}
	}
}

// https://leetcode.cn/problems/container-with-most-water/
// 11. 盛最多水的容器
func maxArea(height []int) int {
	var max int
	for i, j := 0, len(height)-1; i < j; {
		var cur int
		if height[i] < height[j] {
			cur = height[i] * (j - i)
			i++
		} else {
			cur = height[j] * (j - i)
			j--
		}
		if cur > max {
			max = cur
		}
	}
	return max
}

// https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-100-liked
// 15.三数之和
func threeSum(nums []int) [][]int {
	var res [][]int
	sort.Ints(nums)
	for i := 0; i < len(nums)-2 && nums[i] <= 0; i++ {
		if i == 0 || nums[i] != nums[i-1] {
			for j, k := i+1, len(nums)-1; j < k; {
				if nums[i]+nums[j]+nums[k] == 0 {
					res = append(res, []int{nums[i], nums[j], nums[k]})
					j++
					k--
					for j < k && nums[j] == nums[j-1] {
						j++
					}
					for j < k && nums[k] == nums[k+1] {
						k--
					}
				} else if nums[i]+nums[j]+nums[k] < 0 {
					j++
				} else {
					k--
				}
			}
		}
	}
	return res
}

// https://leetcode.cn/problems/trapping-rain-water/?envType=study-plan-v2&envId=top-100-liked
// 42. 接雨水
func trap(height []int) int {
	var res int
	var stack []int
	for i, h := range height {
		for len(stack) > 0 && h > height[stack[len(stack)-1]] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}
			curH := min(h, height[stack[len(stack)-1]]) - height[top]
			curW := i - stack[len(stack)-1] - 1
			res += curH * curW
		}
		stack = append(stack, i)
	}
	return res
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked
// 3. 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	left, right := 0, 0
	res := 0
	mmap := make(map[byte]int)
	for right < len(s) {
		v := s[right]
		right++
		mmap[v]++
		for mmap[v] > 1 {
			mmap[s[left]]--
			left++
		}
		if right-left > res {
			res = right - left
		}
	}
	return res
}

// https://leetcode.cn/problems/find-all-anagrams-in-a-string/?envType=study-plan-v2&envId=top-100-liked
// 438. 找到字符串中所有字母异位词
func findAnagrams(s string, p string) []int {
	var res []int
	var left, right int
	var window [26]int
	var needs [26]int
	for _, v := range p {
		needs[v-'a']++
	}
	for right < len(s) {
		window[s[right]-'a']++
		right++
		for window[s[right-1]-'a'] > needs[s[right-1]-'a'] {
			window[s[left]-'a']--
			left++
		}
		if right-left == len(p) {
			res = append(res, left)
		}
	}
	return res
}
