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

//977. 有序数组的平方
func sortedSquares(nums []int) []int {
	n := len(nums)
	a := make([]int, n)
	left := 0
	right := n - 1
	index := n - 1
	for left <= right {
		if nums[left]*nums[left] > nums[right]*nums[right] {
			a[index] = nums[left] * nums[left]
			left++
		} else {
			a[index] = nums[right] * nums[right]
			right--
		}
		index--
	}
	return a
}

//189. 旋转数组
func rotate(nums []int, k int) {
	k %= len(nums)
	reverse1(nums)
	reverse1(nums[:k])
	reverse1(nums[k:])
}
func reverse1(a []int) {
	for i, n := 0, len(a); i < n/2; i++ {
		a[i], a[n-1-i] = a[n-1-i], a[i]
	}
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

//344. 反转字符串
func reverseString(s []byte) {
	n := len(s)
	for i := 0; i < n/2; i++ {
		s[i], s[n-1-i] = s[n-1-i], s[i]
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

//876. 链表的中间结点
func middleNode(head *ListNode) *ListNode {
	i := head
	j := head
	for j.Next != nil && j.Next.Next != nil {
		j = j.Next.Next
		i = i.Next
	}
	if j.Next != nil {
		i = i.Next
	}
	return i
}

//557. 反转字符串中的单词 III
func reverseWords(s string) string {
	i := 0
	j := 0
	res := ""
	for i < len(s) && j < len(s) {
		if s[j] == ' ' {
			str := reverse(s[i:j])
			res += str + " "
			i = j + 1
		} else if j == len(s)-1 {
			str := reverse(s[i : j+1])
			res += str
		}
		j++
	}
	return res
}
func reverse(s string) string {
	str := []byte(s)
	n := len(str)
	for i := 0; i < n/2; i++ {
		str[i], str[n-1-i] = s[n-1-i], s[i]
	}
	return string(str)
}

//19. 删除链表的倒数第 N 个结点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	if head.Next == nil {
		return nil
	}
	rhead := &ListNode{}
	rhead.Next = head
	p := rhead
	q := rhead
	i := 0
	for q.Next != nil {
		q = q.Next
		if i == n {
			p = p.Next
		} else {
			i++
		}
	}
	p.Next = p.Next.Next
	return rhead.Next
}

//567. 字符串的排列
func checkInclusion(s1 string, s2 string) bool {
	n := len(s1)
	m := len(s2)
	if n > m {
		return false
	}
	cun1 := [26]int{}
	for _, v := range s1 {
		cun1[v-'a']++
	}
	for i := 0; i <= m-n; i++ {
		cun2 := [26]int{}
		for j := i; j < i+n; j++ {
			cun2[s2[j]-'a']++
		}
		if cun1 == cun2 {
			return true
		}
	}
	return false
}

//733. 图像渲染
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	if newColor != image[sr][sc] {
		dfs2(image, sr, sc, newColor, image[sr][sc])
	}
	return image
}
func dfs2(image [][]int, sr int, sc int, newColor, oldColor int) {
	if sr < 0 || sc < 0 || sr >= len(image) || sc >= len(image[sr]) {
		return
	}
	if image[sr][sc] == oldColor {
		image[sr][sc] = newColor
		dfs2(image, sr-1, sc, newColor, oldColor)
		dfs2(image, sr+1, sc, newColor, oldColor)
		dfs2(image, sr, sc-1, newColor, oldColor)
		dfs2(image, sr, sc+1, newColor, oldColor)
	}
}

//695. 岛屿的最大面积
func maxAreaOfIsland(grid [][]int) int {
	area := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				a := dfs3(grid, i, j)
				if a > area {
					area = a
				}
			}
		}
	}
	return area
}
func dfs3(grid [][]int, i int, j int) int {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[i]) {
		return 0
	}
	if grid[i][j] == 1 {
		grid[i][j] = 0
		return 1 + dfs3(grid, i-1, j) + dfs3(grid, i+1, j) + dfs3(grid, i, j-1) + dfs3(grid, i, j+1)
	}
	return 0
}

//15. 三数之和
func threeSum(nums []int) [][]int {
	res := [][]int{}
	n := len(nums)
	if n == 0 {
		return res
	}
	sort.Ints(nums)
	if nums[0] > 0 {
		return res
	}
	for i := 0; i < n; i++ {
		if nums[i] > 0 {
			continue
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		l := i + 1
		r := n - 1
		for r > l {
			if nums[i]+nums[l]+nums[r] == 0 {
				res = append(res, []int{nums[i], nums[l], nums[r]})
				for l < r && nums[l] == nums[l+1] {
					l++
				}
				for l < r && nums[r] == nums[r-1] {
					r--
				}
				l++
				r--
			} else if nums[i]+nums[l]+nums[r] > 0 {
				r--
			} else {
				l++
			}
		}
	}
	return res
}

//617. 合并二叉树
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil {
		return root2
	}
	if root2 == nil {
		return root1
	}
	return dfs4(root1, root2)
}
func dfs4(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil {
		return root2
	}
	if root2 == nil {
		return root1
	}
	root1.Val = root1.Val + root2.Val
	root1.Left = dfs4(root1.Left, root2.Left)
	root1.Right = dfs4(root1.Right, root2.Right)
	return root1
}

//116. 填充每个节点的下一个右侧节点指针
type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	p := root
	for p.Left != nil {
		temp := p
		for temp != nil {
			temp.Left.Next = temp.Right
			if temp.Next != nil {
				temp.Right.Next = temp.Next.Left
			}
			temp = temp.Next
		}
		p = p.Left
	}
	return root
}

//542. 01 矩阵
func updateMatrix(mat [][]int) [][]int {
	q := [][]int{}
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			if mat[i][j] == 0 {
				q = append(q, []int{i, j})
			} else if mat[i][j] == 1 {
				mat[i][j] = -1
			}
		}
	}
	for len(q) > 0 {
		c := q[0]
		q = q[1:]
		if c[0] < len(mat)-1 && mat[c[0]+1][c[1]] == -1 {
			mat[c[0]+1][c[1]] = mat[c[0]][c[1]] + 1
			q = append(q, []int{c[0] + 1, c[1]})
		}
		if c[0] > 0 && mat[c[0]-1][c[1]] == -1 {
			mat[c[0]-1][c[1]] = mat[c[0]][c[1]] + 1
			q = append(q, []int{c[0] - 1, c[1]})
		}
		if c[1] < len(mat[c[0]])-1 && mat[c[0]][c[1]+1] == -1 {
			mat[c[0]][c[1]+1] = mat[c[0]][c[1]] + 1
			q = append(q, []int{c[0], c[1] + 1})
		}
		if c[1] > 0 && mat[c[0]][c[1]-1] == -1 {
			mat[c[0]][c[1]-1] = mat[c[0]][c[1]] + 1
			q = append(q, []int{c[0], c[1] - 1})
		}
	}
	return mat
}

//994. 腐烂的橘子
func orangesRotting(grid [][]int) int {
	q := [][]int{}
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 2 {
				q = append(q, []int{i, j})
			} else if grid[i][j] == 1 {
				count++
			}
		}
	}
	size := 0
	c := [][]int{}
	for len(q) > 0 && count > 0 {
		c = append(c, q[0])
		q = q[1:]
		if len(q) == 0 {
			for _, v := range c {
				if v[0] > 0 && grid[v[0]-1][v[1]] == 1 {
					grid[v[0]-1][v[1]] = 2
					count--
					q = append(q, []int{v[0] - 1, v[1]})
				}
				if v[0] < len(grid)-1 && grid[v[0]+1][v[1]] == 1 {
					grid[v[0]+1][v[1]] = 2
					count--
					q = append(q, []int{v[0] + 1, v[1]})
				}
				if v[1] > 0 && grid[v[0]][v[1]-1] == 1 {
					grid[v[0]][v[1]-1] = 2
					count--
					q = append(q, []int{v[0], v[1] - 1})
				}
				if v[1] < len(grid[v[0]])-1 && grid[v[0]][v[1]+1] == 1 {
					grid[v[0]][v[1]+1] = 2
					count--
					q = append(q, []int{v[0], v[1] + 1})
				}
			}
			size++
		}
	}
	if count > 0 {
		return -1
	}
	return size
}

// 21. 合并两个有序链表
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	l := &ListNode{}
	p := l
	for l1 != nil && l2 != nil {
		n := &ListNode{}
		if l1.Val < l2.Val {
			n = l1
			l1 = l1.Next
		} else {
			n = l2
			l2 = l2.Next
		}
		p.Next = n
		p = p.Next
	}
	if l1 != nil {
		p.Next = l1
	}
	if l2 != nil {
		p.Next = l2
	}
	return l.Next
}

//206. 反转链表
type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
	var p *ListNode
	q := head
	for q != nil {
		n := q.Next
		q.Next = p
		p = q
		q = n
	}
	return p
}

//77. 组合
var res = [][]int{}
var path = []int{}

func combine(n int, k int) [][]int {
	dfs6(1, n, k)
	return res
}
func dfs6(start int, n, k int) {
	if len(path) == k {
		comb := make([]int, k)
		copy(comb, path)
		res = append(res, comb)
		return
	}
	if len(path)+n-start+1 < k {
		return
	}
	for i := start; i <= n; i++ {
		path = append(path, i)
		dfs6(i+1, n, k)
		path = path[:len(path)-1]
	}
}

//46. 全排列
var ans = [][]int{}
var p = []int{}

func permute(nums []int) [][]int {
	n := len(nums)
	used := make([]bool, n)
	dfs7(nums, n, used)
	return ans
}
func dfs7(nums []int, n int, used []bool) {
	if len(p) == n {
		temp := make([]int, n)
		copy(temp, p)
		ans = append(ans, temp)
		return
	}
	for i := 0; i < n; i++ {
		if !used[i] {
			p = append(p, nums[i])
			used[i] = true
			dfs7(nums, n, used)
			p = p[:len(p)-1]
			used[i] = false
		}
	}
}

//784. 字母大小写全排列
var ans1 []string
var as []byte

func letterCasePermutation(s string) []string {
	n := len(s)
	dfs8(s, 0, n)
	return ans1
}
func dfs8(s string, start, n int) {
	if len(as) == n {
		ts := string(as)
		ans1 = append(ans1, ts)
		return
	}
	for i := start; i < len(s); i++ {
		if s[i] >= 'a' && s[i] <= 'z' {
			as = append(as, s[i])
			dfs8(s, i+1, n)
			as = as[:len(as)-1]

			as = append(as, s[i]-32)
			dfs8(s, i+1, n)
			as = as[:len(as)-1]
		} else if s[i] >= 'A' && s[i] <= 'Z' {
			as = append(as, s[i])
			dfs8(s, i+1, n)
			as = as[:len(as)-1]

			as = append(as, s[i]+32)
			dfs8(s, i+1, n)
			as = as[:len(as)-1]
		} else {
			as = append(as, s[i])
			dfs8(s, i+1, n)
			as = as[:len(as)-1]
		}
	}
}

//70. 爬楼梯
func climbStairs(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	c := make([]int, n+1)
	c[1] = 1
	c[2] = 2
	for i := 3; i <= n; i++ {
		c[i] = c[i-1] + c[i-2]
	}
	return c[n]
}

//198. 打家劫舍
func rob(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	dp := make([]int, n)
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < n; i++ {
		dp[i] = max(dp[i-1], nums[i]+dp[i-2])
	}
	return dp[n-1]
}

//120. 三角形最小路径和
func minimumTotal(triangle [][]int) int {
	dp := [][]int{}
	for i := 0; i < len(triangle); i++ {
		d := make([]int, len(triangle[i]))
		dp = append(dp, d)
	}
	return dfs9(triangle, 0, 0, dp)
}

func dfs9(triangle [][]int, i int, j int, dp [][]int) int {
	if i == len(triangle) {
		return 0
	}
	if dp[i][j] != 0 {
		return dp[i][j]
	}
	dp[i][j] = min(dfs9(triangle, i+1, j, dp), dfs9(triangle, i+1, j+1, dp)) + triangle[i][j]
	return dp[i][j]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

//231. 2 的幂
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1) == 0)
}

//191. 位1的个数
func hammingWeight(num uint32) int {
	ones := 0
	for ; num > 0; num &= num - 1 {
		ones++
	}
	return ones
}

//136. 只出现一次的数字
func singleNumber(nums []int) int {
	single := 0
	for _, num := range nums {
		fmt.Println(num)
		single ^= num
		fmt.Println(single)
	}
	return single
}
func main() {
	nums := []int{2, 2, 1}
	//s := "aadasfaf"
	//count := strings.Count(s,"")
	//fmt.Println(count)
	//count = utf8.RuneCountInString(s)
	//fmt.Println(count)
	//fmt.Println(twoSum(nums, 9))
	//s := generateParenthesis(2)
	//fmt.Println(s)
	//s := []int{10, 4, 7, 1, 5, 2}
	//add(s)
	//BumSort(s)
	//QuickSort(s, 0, len(s)-1)
	//res := combine(1,1)
	//res1:=permute(nums)
	//res1 := rob(nums)
	//fmt.Println(res1)
	//
	singleNumber(nums)
}
