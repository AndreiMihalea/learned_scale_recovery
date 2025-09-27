"""
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

 

Example 1:

Input: x = 123
Output: 321
Example 2:

Input: x = -123
Output: -321
Example 3:

Input: x = 120
Output: 21
 

Constraints:

-231 <= x <= 231 - 1
"""
class Solution:
    def reverse(self, x: int) -> int:
        res = 0
        sgn = -1 if x < 0 else 1
        x = abs(x)
        while x:
            last_digit = x % 10
            print(last_digit)
            if (2 ** 31 - 1 - last_digit) // 10 < res:
                return 0
            res = res * 10 + last_digit
            x //= 10
        return res * sgn
        
