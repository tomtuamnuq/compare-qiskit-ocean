\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: test_5

Minimize
 obj: 2 x0_0 + 2 x0_1 - 2 x0_2 + x0_4
Subject To
 A0_leb0: x0_1 + 2 x0_2 - 2 x0_3 + 2 x0_4 <= 3
 A0_leb1: - 3 x0_1 + 2 x0_2 - x0_3 + x0_4 <= -1
 A0_leb2: - x0_0 + 2 x0_1 + 2 x0_2 + 2 x0_3 + 2 x0_4 <= 5

Bounds
 0 <= x0_0 <= 1
 0 <= x0_1 <= 1
 0 <= x0_2 <= 1
 0 <= x0_3 <= 1
 0 <= x0_4 <= 1

Binaries
 x0_0 x0_1 x0_2 x0_3 x0_4
End
