package main

import (
	"fmt"
	"sort"
)

//1. 两数之和
func twoSum(nums []int, target int) []int {
	hashTable := map[int]int{}
	for i, x := range nums {
		if p, ok := hashTable[target-x]; ok {
			return []int{p, i}
		}
		hashTable[x] = i
	}
	return nil
}

//5. 最长回文子串
func longestPalindrome(s string) string {
	start := 0
	end := 0
	for i := 0; i < len(s); i++ {
		l1 := explo(s, i, i)
		l2 := explo(s, i, i+1)
		l := max(l1, l2)
		if l > end-start+1 {
			start = i - l/2
			end = i + l/2
		}
	}
	return s[start : end+1]
}
func explo(s string, start, end int) int {
	for start >= 0 && end < len(s) {
		if s[start] == s[end] {
			start--
			end++
		} else {
			break
		}
	}
	return end - start - 1
}

//3. 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	r := 0
	ans := 0
	m := map[byte]int{}
	for i := 0; i < len(s); i++ {
		if i != 0 {
			delete(m, s[i-1])
		}
		for r < len(s) && m[s[r]] == 0 {
			m[s[r]]++
			r++
		}
		ans = max(ans, r-i)
	}
	return ans
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

//22. 括号生成
func generateParenthesis(n int) []string {
	s := []string{}
	if n == 0 {
		return s
	}
	dfs("", n, n, &s)
	return s
}
func dfs(curStr string, left int, right int, res *[]string) {
	if left == 0 && right == 0 {
		*res = append(*res, curStr)
		return
	}
	if left > right {
		return
	}
	if left > 0 {
		dfs(curStr+"(", left-1, right, res)
	}
	if right > 0 {
		dfs(curStr+")", left, right-1, res)
	}
}
func add(s []int) {
	s = append(s, 1)
	return
}

//704. 二分查找
func search(nums []int, target int) int {
	low := 0
	high := len(nums) - 1
	for low <= high {
		mid := (low + high) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			low = mid + 1
		} else if nums[mid] > target {
			high = mid - 1
		}
	}
	return -1
}

//278. 第一个错误的版本
func firstBadVersion(n int) int {
	low := 1
	high := n
	bad := []int{}
	for low <= high {
		mid := (low + high) / 2
		res := isBadVersion(mid)
		if res {
			bad = append(bad, mid)
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	sort.Ints(bad)
	return bad[0]
}
func isBadVersion(version int) bool {
	if version == 2 {
		return true
	}
	return false
}

//35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	low := 0
	high := len(nums) - 1
	mid := (low + high) / 2
	for low <= high {
		mid = (low + high) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			low = mid + 1
		} else if nums[mid] > target {
			high = mid - 1
		}
	}
	if high < 0 {
		return 0
	}
	if low > len(nums)-1 {
		return len(nums)
	}
	return low
}
func BumSort(a []int) {
	for i := 0; i < len(a)-1; i++ {
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[i] {
				a[j], a[i] = a[i], a[j]
			}
		}
	}
}
func QuickSort(a []int, left, right int) {
	if left > right {
		return
	}
	i := left
	j := right
	flag := a[left]
	for i < j {
		for a[j] >= flag && i < j {
			j--
		}
		for a[i] <= flag && i < j {
			i++
		}
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i], a[left] = a[left], a[i]
	QuickSort(a, left, j-1)
	QuickSort(a, j+1, right)
}

//283. 移动零
func moveZeroes(nums []int) {
	j := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[j] = nums[i]
			j++
		}
	}
	for i := j; i < len(nums); i++ {
		nums[i] = 0
	}
}

//167. 两数之和 II - 输入有序数组
func twoSum2(numbers []int, target int) []int {
	i := 0
	j := len(numbers) - 1
	for i < j {
		sum := numbers[i] + numbers[j]
		if sum == target {
			return []int{i + 1, j + 1}
		} else if sum < target {
			i++
		} else {
			j--
		}
	}
	return []int{}
}
func main() {
	//nums := []int{2, 7, 11, 15}
	//fmt.Println(twoSum(nums, 9))
	//s := generateParenthesis(2)
	//fmt.Println(s)
	s := []int{10, 4, 7, 1, 5, 2}
	//add(s)
	//BumSort(s)
	QuickSort(s, 0, len(s)-1)
	fmt.Println(s)
}
