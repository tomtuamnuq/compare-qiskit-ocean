\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: test_4

Minimize
 obj: 3 x_0 + x_1 + 5 x_2 - 3 x1_0 + 4 x1_1 - 5 x1_2 + 2 x2_0 - 5 x2_1 - 2 x2_2
      - x3_0 + 3 x3_2 + [ 10 x_0^2 + 14 x_0*x_1 + 4 x_1^2 + 4 x_1*x_2 + 10 x_2^2
      - 10 x1_0^2 + 2 x1_0*x1_1 + 8 x1_0*x1_2 - 8 x1_1^2 + 6 x1_1*x1_2
      - 2 x2_0^2 - 10 x2_0*x2_1 + 6 x2_0*x2_2 + 8 x2_1*x2_2 + 8 x2_2^2
      + 6 x3_0^2 + 2 x3_0*x3_1 - 10 x3_0*x3_2 + 10 x3_1^2 + 2 x3_1*x3_2
      - 10 x3_2^2 ]/2
Subject To

Bounds
 0 <= x_0 <= 1
 0 <= x_1 <= 1
 0 <= x_2 <= 1
 0 <= x1_0 <= 1
 0 <= x1_1 <= 1
 0 <= x1_2 <= 1
 0 <= x2_0 <= 1
 0 <= x2_1 <= 1
 0 <= x2_2 <= 1
 0 <= x3_0 <= 1
 0 <= x3_1 <= 1
 0 <= x3_2 <= 1

Binaries
 x_0 x_1 x_2 x1_0 x1_1 x1_2 x2_0 x2_1 x2_2 x3_0 x3_1 x3_2
End
