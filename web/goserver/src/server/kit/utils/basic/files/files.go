package files

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"
)

func Copy(src, dst string) (int64, error) {
	src = fp.Clean(src)
	dst = fp.Clean(dst)

	in, err := os.Open(src)
	if err != nil {
		log.Println("files.Copy.os.Open(src)", err)
		return 0, err
	}
	defer in.Close()

	err = os.MkdirAll(fp.Dir(dst), 0777)
	if err != nil {
		log.Println("files.Copy.os.MkdirAll(fp.Dir(dst)", err)
		return 0, err
	}

	out, err := os.Create(dst)
	if err != nil {
		log.Println("files.Copy.os.Create(dst)", err)
		return 0, err
	}
	defer func() {
		if e := out.Close(); e != nil {
			log.Println("files.Copy.out.Close()", err)
			err = e
		}
	}()

	nBytes, err := io.Copy(out, in)
	if err != nil {
		log.Println("files.Copy.io.Copy(out, in)", err)
		return nBytes, err
	}

	err = out.Sync()
	if err != nil {
		log.Println("files.Copy.out.Sync()", err)
		return nBytes, err
	}

	si, err := os.Stat(src)
	if err != nil {
		log.Println("files.Copy.os.Stat(src)", err)
		return nBytes, err
	}
	err = os.Chmod(dst, si.Mode())
	if err != nil {
		log.Println("files.Copy.os.Chmod(dst, si.Mode())", err)
		return nBytes, err
	}

	return nBytes, err
}

func CopyDir(src string, dst string) (err error) {
	src = fp.Clean(src)
	dst = fp.Clean(dst)

	si, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !si.IsDir() {
		return fmt.Errorf("source is not a directory")
	}

	_, err = os.Stat(dst)
	if err != nil && !os.IsNotExist(err) {
		return
	}
	if err == nil {
		return fmt.Errorf("destination already exists")
	}

	err = os.MkdirAll(dst, si.Mode())
	if err != nil {
		return
	}

	entries, err := ioutil.ReadDir(src)
	if err != nil {
		return
	}

	for _, entry := range entries {
		srcPath := fp.Join(src, entry.Name())
		dstPath := fp.Join(dst, entry.Name())

		if entry.IsDir() {
			err = CopyDir(srcPath, dstPath)
			if err != nil {
				return
			}
		} else {
			// Skip symlinks.
			if entry.Mode()&os.ModeSymlink != 0 {
				continue
			}

			_, err = Copy(srcPath, dstPath)
			if err != nil {
				return
			}
		}
	}

	return
}
