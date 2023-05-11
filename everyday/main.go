package main

import (
	"strconv"
	"strings"
)

// 2023.4.26
// https://leetcode.cn/problems/maximum-sum-of-two-non-overlapping-subarrays/
func maxSumTwoNoOverlap(nums []int, firstLen int, secondLen int) int {
	// TODO
	return 0
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// https://leetcode.cn/problems/moving-stones-until-consecutive/
func numMovesStones(a int, b int, c int) []int {
	x := min(min(a, b), c)
	z := max(max(a, b), c)
	y := a + b + c - x - z
	res := []int{2, z - x - 2}
	if z-y == 1 && y-x == 1 {
		res[0] = 0
	} else if z-y <= 2 || y-x <= 2 {
		res[0] = 1
	}
	return res
}

// https://leetcode.cn/problems/time-needed-to-inform-all-employees/
func numOfMinutes(n int, headID int, manager []int, informTime []int) int {
	g := make(map[int][]int)
	for i, m := range manager {
		g[m] = append(g[m], i)
	}
	var dfs func(int) int
	dfs = func(cur int) (res int) {
		for _, neighbor := range g[cur] {
			res1 := dfs(neighbor)
			if res1 > res {
				res = res1
			}
		}
		return informTime[cur] + res
	}
	return dfs(headID)
}

// https://leetcode.cn/problems/binary-string-with-substrings-representing-1-to-n/
func queryString(S string, N int) bool {
	for i := 1; i <= N; i++ {
		if !strings.Contains(S, strconv.FormatInt(int64(i), 2)) {
			return false
		}
	}
	return true
}
