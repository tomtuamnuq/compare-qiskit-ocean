\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: test_22

Minimize
 obj: x0_1 + x0_2 + 2 x1_0 + 2 x1_1 + 2 x1_2 - x2_0 + 2 x2_1 + 2 x3_1 - 2 x3_2
      - x4_1 - 2 x4_2 + x5_0 + x5_1 + 2 x5_2 + 2 x6_0 + x6_1 + x7_0 + x7_1
      - x7_2 + 2 x8_2 + 2 x9_0 - 2 x9_1 + x9_2 - x10_0 - 2 x10_2 + 2 x12_0
      - 2 x12_1 - 2 x12_2 + x13_1 - x13_2 - x14_0 + 2 x14_1 + x14_2 + x15_0
      - x15_1 + 2 x15_2 - 2 x16_0 - x16_1 - x16_2 + 2 x17_0 + 2 x17_1 + x17_2
      - x18_0 - x18_1 - x19_0 + x19_1 - 2 x19_2 - x20_0 + x20_1 + 2 x20_2
      - x21_0 - x21_1
Subject To
 A0_leb0: x0_0 + 2 x0_1 + 3 x0_2 <= 7
 A0_leb1: - x0_0 - 2 x0_1 - x0_2 <= -3
 A1_leb0: - 3 x1_0 + 2 x1_1 + 2 x1_2 <= 1
 A1_leb1: - 2 x1_0 + 2 x1_1 + 3 x1_2 <= 1
 A2_leb0: - x2_0 - 3 x2_1 - 2 x2_2 <= 0
 A2_leb1: 2 x2_0 - 3 x2_2 <= 0
 A3_leb0: - 2 x3_0 + 3 x3_1 + x3_2 <= 2
 A3_leb1: x3_0 + 2 x3_1 - 2 x3_2 <= 0
 A4_leb0: x4_1 - 3 x4_2 <= -3
 A4_leb1: - 3 x4_0 - 2 x4_1 + 2 x4_2 <= 3
 A5_leb0: x5_0 + 2 x5_2 <= 1
 A5_leb1: - 2 x5_0 + 2 x5_1 - x5_2 <= 3
 A6_leb0: - 2 x6_0 - x6_1 + 3 x6_2 <= -2
 A6_leb1: 3 x6_0 - 3 x6_1 + x6_2 <= 5
 A7_leb0: - 3 x7_0 + 2 x7_2 <= 1
 A7_leb1: - 2 x7_0 + 3 x7_1 + x7_2 <= 0
 A8_leb0: x8_1 - x8_2 <= 0
 A8_leb1: 3 x8_0 - 3 x8_2 <= 3
 A9_leb0: 2 x9_0 - x9_1 + 3 x9_2 <= 1
 A9_leb1: 3 x9_0 - 2 x9_1 + 3 x9_2 <= 0
 A10_leb0: - x10_0 - 2 x10_1 - 3 x10_2 <= -1
 A10_leb1: - x10_1 + 2 x10_2 <= 1
 A11_leb0: - 3 x11_0 + x11_1 <= 2
 A11_leb1: 2 x11_0 + 2 x11_1 + 2 x11_2 <= 2
 A12_leb0: 3 x12_0 - x12_1 <= 1
 A12_leb1: - 2 x12_0 - 3 x12_1 - 3 x12_2 <= -2
 A13_leb0: - 2 x13_0 - 2 x13_2 <= -4
 A13_leb1: 3 x13_0 + 3 x13_1 + x13_2 <= 6
 A14_leb0: 2 x14_0 + x14_1 - 3 x14_2 <= 3
 A14_leb1: - 2 x14_0 + 3 x14_1 + 3 x14_2 <= 0
 A15_leb0: - 3 x15_1 - 2 x15_2 <= -1
 A15_leb1: 2 x15_0 + 3 x15_1 - 2 x15_2 <= 7
 A16_leb0: x16_0 - 2 x16_1 - 2 x16_2 <= -1
 A16_leb1: 3 x16_0 + 2 x16_1 + 2 x16_2 <= 7
 A17_leb0: - 3 x17_0 - 3 x17_1 - 3 x17_2 <= -6
 A17_leb1: 2 x17_0 + x17_2 <= 1
 A18_leb0: x18_0 - 3 x18_1 <= -1
 A18_leb1: 3 x18_0 - x18_2 <= 3
 A19_leb0: x19_0 - x19_2 <= 2
 A19_leb1: - x19_0 + 3 x19_1 - x19_2 <= 1
 A20_leb0: - 3 x20_0 + 3 x20_1 + 2 x20_2 <= 1
 A20_leb1: - x20_0 + 2 x20_1 + x20_2 <= 1
 A21_leb0: 3 x21_0 + 3 x21_1 - 2 x21_2 <= 0
 A21_leb1: x21_0 - x21_1 + 3 x21_2 <= 0

Bounds
 0 <= x0_0 <= 1
 0 <= x0_1 <= 1
 0 <= x0_2 <= 1
 0 <= x1_0 <= 1
 0 <= x1_1 <= 1
 0 <= x1_2 <= 1
 0 <= x2_0 <= 1
 0 <= x2_1 <= 1
 0 <= x2_2 <= 1
 0 <= x3_0 <= 1
 0 <= x3_1 <= 1
 0 <= x3_2 <= 1
 0 <= x4_0 <= 1
 0 <= x4_1 <= 1
 0 <= x4_2 <= 1
 0 <= x5_0 <= 1
 0 <= x5_1 <= 1
 0 <= x5_2 <= 1
 0 <= x6_0 <= 1
 0 <= x6_1 <= 1
 0 <= x6_2 <= 1
 0 <= x7_0 <= 1
 0 <= x7_1 <= 1
 0 <= x7_2 <= 1
 0 <= x8_0 <= 1
 0 <= x8_1 <= 1
 0 <= x8_2 <= 1
 0 <= x9_0 <= 1
 0 <= x9_1 <= 1
 0 <= x9_2 <= 1
 0 <= x10_0 <= 1
 0 <= x10_1 <= 1
 0 <= x10_2 <= 1
 0 <= x11_0 <= 1
 0 <= x11_1 <= 1
 0 <= x11_2 <= 1
 0 <= x12_0 <= 1
 0 <= x12_1 <= 1
 0 <= x12_2 <= 1
 0 <= x13_0 <= 1
 0 <= x13_1 <= 1
 0 <= x13_2 <= 1
 0 <= x14_0 <= 1
 0 <= x14_1 <= 1
 0 <= x14_2 <= 1
 0 <= x15_0 <= 1
 0 <= x15_1 <= 1
 0 <= x15_2 <= 1
 0 <= x16_0 <= 1
 0 <= x16_1 <= 1
 0 <= x16_2 <= 1
 0 <= x17_0 <= 1
 0 <= x17_1 <= 1
 0 <= x17_2 <= 1
 0 <= x18_0 <= 1
 0 <= x18_1 <= 1
 0 <= x18_2 <= 1
 0 <= x19_0 <= 1
 0 <= x19_1 <= 1
 0 <= x19_2 <= 1
 0 <= x20_0 <= 1
 0 <= x20_1 <= 1
 0 <= x20_2 <= 1
 0 <= x21_0 <= 1
 0 <= x21_1 <= 1
 0 <= x21_2 <= 1

Binaries
 x0_0 x0_1 x0_2 x1_0 x1_1 x1_2 x2_0 x2_1 x2_2 x3_0 x3_1 x3_2 x4_0 x4_1 x4_2 x5_0
 x5_1 x5_2 x6_0 x6_1 x6_2 x7_0 x7_1 x7_2 x8_0 x8_1 x8_2 x9_0 x9_1 x9_2 x10_0
 x10_1 x10_2 x11_0 x11_1 x11_2 x12_0 x12_1 x12_2 x13_0 x13_1 x13_2 x14_0 x14_1
 x14_2 x15_0 x15_1 x15_2 x16_0 x16_1 x16_2 x17_0 x17_1 x17_2 x18_0 x18_1 x18_2
 x19_0 x19_1 x19_2 x20_0 x20_1 x20_2 x21_0 x21_1 x21_2
End
