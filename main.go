package main

import "fmt"

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
func main() {
	//nums := []int{2, 7, 11, 15}
	//fmt.Println(twoSum(nums, 9))
	//s := generateParenthesis(2)
	//fmt.Println(s)
	s := []int{}
	add(s)
	fmt.Println(s)
}
