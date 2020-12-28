input = [3, 4, 5, 6, 7]
index = [1 2 3 4;
         4 2 1 3;
         3 5 5 3]
output = [3 4 5 6;
          6 4 3 5;
          5 7 7 5]
@test gather(input, index) == output
