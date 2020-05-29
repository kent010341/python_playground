import numpy as np
import random
from psoada import pso

city_count = 50
loc_ub, loc_lb = 50, -50

np.random.seed(0)
random.seed(0)

def cost_func(params):
	global dist_arr

	lehmer_code = np.round(params).tolist() + [0]
	cities_seq = lehmer_decode(lehmer_code)
	cities_seq += [cities_seq[0]]

	dist = 0
	for i in range(len(cities_seq)-1):
		start, end = cities_seq[i], cities_seq[i+1]
		dist += dist_arr[start, end]

	return dist

def lehmer_decode(lehmer_code):
	temp_seq = list(range(city_count))
	opt = list()
	for code in lehmer_code:
		temp = temp_seq[int(code)]
		opt.append(temp)
		temp_seq.remove(temp)

	return opt

def generating_cities():
	global loc_cities, dist_arr

	loc_cities = np.random.uniform(loc_lb, loc_ub, (city_count, 2))
	dist_arr = np.zeros((city_count, city_count))
	for i in range(city_count):
		for j in range(i, city_count):
			if i == j:
				dist_arr[i, j] = 0
			else:
				dist_arr[i, j] = np.sqrt(np.square(loc_cities[i] - loc_cities[j]).sum())
				dist_arr[j, i] = dist_arr[i, j]
	print(dist_arr)

def main():
	generating_cities()

	ub = list(range(city_count-1, 0, -1))
	lb = np.zeros(city_count-1).tolist()

	pso_opt = pso(cost_func, lb, ub, swarmsize=200, debug=False)
	opt_lehmer = pso_opt[0]
	print(opt_lehmer)
	print(lehmer_decode(np.round(opt_lehmer).tolist() + [0]))
	print(cost_func(opt_lehmer))

if __name__ == '__main__':
	main()