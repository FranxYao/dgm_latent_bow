"""Compare the outputs of different models"""

output_file1 = '../outputs/latent_bow_0.5.0.0/output_e6.txt'
output_file2 = '../outputs/seq2seq_0.6.0.0/output_e6.txt'

outputs1 = open(output_file1, 'r').readlines()
outputs2 = open(output_file2, 'r').readlines()
num_cases = len(outputs1) // 6

compare_fd = open('compare_results.txt', 'w') 
for i in range(num_cases):
  src = outputs1[6 * i + 1]
  pred1 = outputs1[6 * i + 3]
  pred2 = outputs2[6 * i + 3]
  ref = outputs1[6 * i + 5]
  # assert(outputs1[3 * i] == outputs2[3 * i])
  if(pred1 != pred2):
    compare_fd.write('src:   ' + src)
    compare_fd.write('lbow:  ' + pred1)
    compare_fd.write('s2s:   ' + pred2)
    compare_fd.write('ref:   ' + ref)
    compare_fd.write('----\n')
