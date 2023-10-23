def initialize():
	try:
		import mmpretrain
	except ModuleNotFoundError:
		print('Initializing zoo...')
		install_mim_dependencies()


def install_mim_dependencies():
	mim_dep = [
		'mmpretrain>=1.0.0rc8',
		'mmyolo==0.6.0',
	]

	print('Checking zoo dependencies, please wait...')
	import mim
	mim.install(mim_dep)
	print('Check over')
