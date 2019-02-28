# pylint: skip-file
import struct
import numpy as np

cv_type_to_dtype = {
	5 : np.dtype('float32'),
	6 : np.dtype('float64')
}

dtype_to_cv_type = {v : k for k,v in cv_type_to_dtype.items()}

def write_mat(f, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
    f.write(header)
    f.write(m.data)


def read_mat(f):
	"""
	Reads an OpenCV mat from the given file opened in binary mode
	"""
	rows, cols, stride, type_ = struct.unpack('iiii', f.read(4*4))
	mat = np.fromstring(f.read(rows*stride),dtype=cv_type_to_dtype[type_])
	return mat.reshape(rows,cols)

def read_mkl_vec(f):
	"""
	Reads an OpenCV mat from the given file opened in binary mode
	"""
	# Read past the header information
	f.read(4*4)

	length, stride, type_ = struct.unpack('iii', f.read(3*4))
	mat = np.fromstring(f.read(length*4),dtype=np.float32)
	return mat

def load_mkl_vec(filename):
	"""
	Reads a OpenCV Mat from the given filename
	"""
	return read_mkl_vec(open(filename,'rb'))

def load_mat(filename):
	"""
	Reads a OpenCV Mat from the given filename
	"""
	return read_mat(open(filename,'rb'))

def save_mat(filename, m):
    """Saves mat m to the given filename"""
    return write_mat(open(filename,'wb'), m)

def main():
	f = open('1_to_0.bin','rb')
	vx = read_mat(f)
	vy = read_mat(f)

if __name__ == '__main__':
	main()
