# import shelve

# T='Hiya'
# val=[1,2,3]

# filename='/tmp/shelve.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new

# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()

# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

# print(T)
# # Hiya
# print(val)

import dill                            #pip install dill --user
filename = 'globalsave.pkl'
k = 2
dill.dump_session(filename)

dill.load_session(filename)
