\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: test_10

Minimize
 obj: x0_0 - 2 x0_1 + x0_2 - x0_3 - 2 x0_4 + 2 x0_6 + x0_7 - 2 x0_8 + x0_9
Subject To
 A0_leb0: - x0_0 - x0_1 + x0_2 + 2 x0_3 - 3 x0_4 - 3 x0_5 - 2 x0_6 + 2 x0_7
          + 3 x0_8 + 2 x0_9 <= 4
 A0_leb1: - x0_0 - x0_1 + 2 x0_2 + 3 x0_3 + x0_4 - 2 x0_6 - x0_7 + 3 x0_8
          - 2 x0_9 <= -1
 A0_leb2: - 2 x0_0 - 2 x0_1 - 3 x0_2 + x0_3 + 2 x0_5 + x0_6 - 3 x0_7 + 2 x0_8
          + 3 x0_9 <= 3
 A0_leb3: x0_0 - 3 x0_1 + 3 x0_3 - 2 x0_5 - 3 x0_6 - x0_7 + x0_8 - 2 x0_9 <= -1
 A0_leb4: 3 x0_0 + 3 x0_1 - 3 x0_2 + x0_3 - x0_4 + 3 x0_7 + 2 x0_8 - 2 x0_9 <= 
          8

Bounds
 0 <= x0_0 <= 1
 0 <= x0_1 <= 1
 0 <= x0_2 <= 1
 0 <= x0_3 <= 1
 0 <= x0_4 <= 1
 0 <= x0_5 <= 1
 0 <= x0_6 <= 1
 0 <= x0_7 <= 1
 0 <= x0_8 <= 1
 0 <= x0_9 <= 1

Binaries
 x0_0 x0_1 x0_2 x0_3 x0_4 x0_5 x0_6 x0_7 x0_8 x0_9
End
