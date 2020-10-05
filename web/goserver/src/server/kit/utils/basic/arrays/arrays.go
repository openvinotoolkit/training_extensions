package arrays

func ContainsString(array []string, item string) bool {
	for _, el := range array {
		if item == el {
			return true
		}
	}
	return false
}

func ContainsInt(array []int, item int) bool {
	for _, el := range array {
		if item == el {
			return true
		}
	}
	return false
}

func FilterInt(arr []int, cond func(int) bool) []int {
	var result []int
	for i := range arr {
		if cond(arr[i]) {
			result = append(result, arr[i])
		}
	}
	return result
}

func MinInt(arr []int) int {
	var m int
	for i, e := range arr {
		if i == 0 || e < m {
			m = e
		}
	}
	return m
}
