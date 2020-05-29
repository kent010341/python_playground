import numpy as np
import matplotlib.pyplot as plt
from CodingToolBox import var_form

#__move_keys__ = ('up', 'down', 'left', 'right')
class AttrMain:
	size = 25
	start_direction = 2 # down
	is_clockwise = True

class AttrMove:
	move = np.array([[-1,  0],  # up
					 [ 1,  0],  # down
					 [ 0, -1],  # left
					 [ 0,  1]]) # right
	#move_str = np.array(['up', 'down', 'left', 'right'])
	if AttrMain.is_clockwise:
		move_seq = np.array([0, 2, 1, 3], dtype=np.int32)
	else:
		move_seq = np.array([0, 3, 1, 2], dtype=np.int32)
	move = move[move_seq]
	move_str = move_str[move_seq]

def main():
	number = 0
	width = 1
	turn_times = 0
	spiral_arr = np.zeros((AttrMain.size, AttrMain.size), dtype=np.int32)
	current_index = np.ones(2, dtype=np.int32)*(int(AttrMain.size/2))
	curr_move = AttrMain.start_direction
	is_break = False

	while not is_break:
		for i in range(width): # in one line
			try:
				spiral_arr[current_index[0], current_index[1]] = number # set value of current index
			except:
				is_break = True
				break
			current_index += AttrMove.move[curr_move] # move by current direction
			number += 1

		curr_move = (curr_move + 1) % 4 # turn
		#print('turn {}'.format(AttrMove.move_str[curr_move]))
		turn_times += 1 
		if turn_times >= 2:
			turn_times = 0
			width += 1

	print(var_form(spiral_arr))
	plt.imshow(spiral_arr, cmap='gray')
	plt.show()

if __name__ == '__main__':
	main()
