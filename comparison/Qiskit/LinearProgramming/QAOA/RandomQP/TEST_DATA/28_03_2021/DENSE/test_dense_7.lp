\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: test_dense_7

Minimize
 obj: x0_3 - x0_4 - x0_6 + [ - 2 x0_0^2 - 4 x0_0*x0_2 + 4 x0_0*x0_4
      + 4 x0_0*x0_5 + 4 x0_0*x0_6 + 2 x0_1*x0_2 - 4 x0_1*x0_6 - 2 x0_2^2
      - 2 x0_2*x0_3 - 2 x0_2*x0_4 + 4 x0_2*x0_5 + 2 x0_2*x0_6 - 2 x0_3^2
      + 4 x0_3*x0_5 + 2 x0_3*x0_6 + 2 x0_4^2 - 4 x0_4*x0_6 + 2 x0_5^2
      - 2 x0_5*x0_6 - 2 x0_6^2 ]/2
Subject To
 A0_leb0: - x0_0 - x0_2 + x0_3 + x0_4 - x0_5 - x0_6 <= -1
 A0_leb1: x0_0 + x0_3 + x0_4 + x0_5 <= 5
 A0_leb2: - x0_0 - x0_1 - x0_3 <= -3

Bounds
 0 <= x0_0 <= 1
 0 <= x0_1 <= 1
 0 <= x0_2 <= 1
 0 <= x0_3 <= 1
 0 <= x0_4 <= 1
 0 <= x0_5 <= 1
 0 <= x0_6 <= 1

Binaries
 x0_0 x0_1 x0_2 x0_3 x0_4 x0_5 x0_6
End
