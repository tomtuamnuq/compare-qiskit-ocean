\ Two by Four polyomino system

Minimize
  obj: 1
Subject To
  cell1: q1                                    + r1                + s1 + s2 + s3                          = 1
  cell2:      q2                                    + r2           + s1 + s2      + s4                     = 1
  cell3:           q3                          + r1      + r3      + s1      + s3      + s5 + s6 + s7      = 1
  cell4:                q4                          + r2      + r4      + s2      + s4 + s5 + s6      + s8 = 1
  cell5:                     q5                + r1      + r3      + s1      + s3 + s4 + s5      + s7      = 1
  cell6:                          q6                + r2      + r4      + s2 + s3 + s4      + s6      + s8 = 1
  cell7:                               q7                + r3                          + s5      + s7 + s8 = 1
  cell8:                                    q8                + r4                          + s6 + s7 + s8 = 1
  count1: q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8                                                             = 1
  count2:                                         r1 + r2 + r3 + r4                                         = 1
  count3:                                                             s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 = 1
Binary
  q1 q2 q3 q4 q5 q6 q7 q8 r1 r2 r3 r4 s1 s2 s3 s4 s5 s6 s7 s8
End