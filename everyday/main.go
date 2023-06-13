package main

import (
	"fmt"
	"math"
	"sort"
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

// https://leetcode.cn/problems/distant-barcodes/
func rearrangeBarcodes(barcodes []int) []int {
	mmap := make(map[int]int)
	for _, v := range barcodes {
		mmap[v]++
	}
	var res []int
	for len(mmap) > 0 {
		var max, maxK int
		for k, v := range mmap {
			if v > max {
				max = v
				maxK = k
			}
		}
		delete(mmap, maxK)
		for i := 0; i < max; i++ {
			res = append(res, maxK)
		}
	}
	return res
}

// https://leetcode.cn/problems/flip-columns-for-maximum-number-of-equal-rows/
func maxEqualRowsAfterFlips(matrix [][]int) int {
	mmap := make(map[string]int)
	for _, row := range matrix {
		var s []int
		for _, v := range row {
			s = append(s, v^row[0])
		}
		var str strings.Builder
		for _, v := range s {
			str.WriteString(strconv.Itoa(v))
		}
		mmap[str.String()]++
	}
	var res int
	for _, v := range mmap {
		if v > res {
			res = v
		}
	}
	return res
}

// https://leetcode.cn/problems/statistics-from-a-large-sample/
// 输入：count = [0,1,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// 输出：[1.00000,3.00000,2.37500,2.50000,3.00000]
func sampleStats(count []int) []float64 {
	n := len(count)
	total := 0
	for i := 0; i < n; i++ {
		total += count[i]
	}
	mean := 0.0
	median := 0.0
	minimum := 256
	maxnum := 0
	mode := 0

	left := (total + 1) / 2
	right := (total + 2) / 2
	cnt := 0
	maxfreq := 0
	sum := 0
	for i := 0; i < n; i++ {
		sum += int(count[i]) * int(i)
		if count[i] > maxfreq {
			maxfreq = count[i]
			mode = i
		}
		if count[i] > 0 {
			if minimum == 256 {
				minimum = i
			}
			maxnum = i
		}
		if cnt < right && cnt+count[i] >= right {
			median += float64(i)
		}
		if cnt < left && cnt+count[i] >= left {
			median += float64(i)
		}
		cnt += count[i]
	}
	mean = float64(sum) / float64(total)
	median = median / 2.0
	return []float64{float64(minimum), float64(maxnum), mean, median, float64(mode)}
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// https://leetcode.cn/problems/delete-nodes-and-return-forest/
func delNodes(root *TreeNode, to_delete []int) []*TreeNode {
	mmap := make(map[int]bool)
	for _, v := range to_delete {
		mmap[v] = true
	}
	var res []*TreeNode
	var dfs func(*TreeNode, bool) *TreeNode
	dfs = func(node *TreeNode, isRoot bool) *TreeNode {
		if node == nil {
			return nil
		}
		deleted := false
		if mmap[node.Val] {
			deleted = true
		}
		if isRoot && !deleted {
			res = append(res, node)
		}
		node.Left = dfs(node.Left, deleted)
		node.Right = dfs(node.Right, deleted)
		if deleted {
			return nil
		}
		return node
	}
	dfs(root, true)
	return res
}

// https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/
func mctFromLeafValues(arr []int) int {
	n := len(arr)
	var res int
	var stack []int
	stack = append(stack, math.MaxInt32)
	for i := 0; i < n; i++ {
		for stack[len(stack)-1] <= arr[i] {
			tmp := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res += tmp * min(stack[len(stack)-1], arr[i])
		}
		stack = append(stack, arr[i])
	}
	for len(stack) > 2 {
		tmp := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res += tmp * stack[len(stack)-1]
	}
	return res
}

// https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/
func maximumTastiness(price []int, k int) int {
	sort.Ints(price)
	left, right := 0, price[len(price)-1]-price[0]
	for left < right {
		mid := (left + right + 1) / 2
		if check(price, k, mid) {
			left = mid
		} else {
			right = mid - 1
		}
	}
	return left
}

func check(price []int, k int, tastiness int) bool {
	prev := int(math.Inf(-1)) >> 1
	cnt := 0
	for _, p := range price {
		if p-prev >= tastiness {
			cnt++
			prev = p
		}
	}
	return cnt >= k
}

// https://leetcode.cn/problems/count-vowel-strings-in-ranges/
func vowelStrings(words []string, queries [][]int) []int {
	n := len(words)
	prefixSums := make([]int, n+1)
	for i := 0; i < n; i++ {
		value := 0
		if checkStringHasPrefixAndSuffixVowel(words[i]) {
			value = 1
		}
		prefixSums[i+1] = prefixSums[i] + value
	}
	ans := make([]int, len(queries))
	for i := 0; i < len(queries); i++ {
		start := queries[i][0]
		end := queries[i][1]
		ans[i] = prefixSums[end+1] - prefixSums[start]
	}
	return ans
}

func checkStringHasPrefixAndSuffixVowel(w string) bool {
	if len(w) == 0 {
		return false
	}
	mmap := make(map[byte]bool)
	mmap['a'] = true
	mmap['e'] = true
	mmap['i'] = true
	mmap['o'] = true
	mmap['u'] = true
	return mmap[w[0]] && mmap[w[len(w)-1]]
}

// https://leetcode.cn/problems/apply-operations-to-an-array/
func applyOperations(nums []int) []int {
	n := len(nums)
	for i := 0; i < n-1; i++ {
		if nums[i] == nums[i+1] {
			nums[i] = nums[i] * 2
			nums[i+1] = 0
		}
	}
	res := make([]int, n)
	j := 0
	for i := 0; i < n; i++ {
		if nums[i] != 0 {
			res[j] = nums[i]
			j++
		}
	}
	return res
}

// https://leetcode.cn/problems/number-of-unequal-triplets-in-array/
func unequalTriplets(nums []int) int {
	total := 0
	mmap := make(map[int]int)
	for _, num := range nums {
		mmap[num]++
	}
	if len(mmap) <= 2 {
		return 0
	}
	n := len(nums)
	t := 0
	for _, v := range mmap {
		total, t = total+t*v*(n-t-v), t+v
	}
	return total
}

func main() {
	//maxEqualRowsAfterFlips([][]int{{0, 0, 0}, {0, 0, 1}, {1, 1, 0}})
	words := []string{"bzmxvzjxfddcuznspdcbwiojiqf", "mwguoaskvramwgiweogzulcinycosovozppl", "uigevazgbrddbcsvrvnngfrvkhmqszjicpieahs", "uivcdsboxnraqpokjzaayedf", "yalc", "bbhlbmpskgxmxosft", "vigplemkoni", "krdrlctodtmprpxwditvcps", "gqjwokkskrb", "bslxxpabivbvzkozzvdaykaatzrpe", "qwhzcwkchluwdnqjwhabroyyxbtsrsxqjnfpadi", "siqbezhkohmgbenbkikcxmvz", "ddmaireeouzcvffkcohxus", "kjzguljbwsxlrd", "gqzuqcljvcpmoqlnrxvzqwoyas", "vadguvpsubcwbfbaviedr", "nxnorutztxfnpvmukpwuraen", "imgvujjeygsiymdxp", "rdzkpk", "cuap", "qcojjumwp", "pyqzshwykhtyzdwzakjejqyxbganow", "cvxuskhcloxykcu", "ul", "axzscbjajazvbxffrydajapweci"}
	queries := [][]int{{4, 4}, {6, 17}, {10, 17}, {9, 18}, {17, 22}, {5, 23}, {2, 5}, {17, 21}, {5, 17}, {4, 8}, {7, 17}, {16, 19}, {7, 12}, {9, 20}, {13, 23}, {1, 5}, {19, 19}}
	fmt.Println(vowelStrings(words, queries))
}
