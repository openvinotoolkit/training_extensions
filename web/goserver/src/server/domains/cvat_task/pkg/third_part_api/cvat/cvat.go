package cvat

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	fp "path/filepath"
	"time"

	t "server/db/pkg/types"
)

func PrepareAnnotation(annotation t.CvatAnnotation) (status int, err error) {
	log.Println("cvat.PrepareAnnotation.Started", err)
	defer log.Println("cvat.PrepareAnnotation.Finished", err)
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, AnnotationUrl)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.PrepareAnnotation.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	request.URL.RawQuery = q.Encode()
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.PrepareAnnotation.client.Do(request)", err)
		return 0, err
	}
	return response.StatusCode, nil
}

func PrepareDataset(annotation t.CvatAnnotation) (status int, err error) {
	log.Println("cvat.PrepareDataset.Started", err)
	defer log.Println("cvat.PrepareDataset.Finished", err)
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, DatasetUrl)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.PrepareDataset.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	request.URL.RawQuery = q.Encode()
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.PrepareDataset.client.Do(request)", err)
		return 0, err
	}
	return response.StatusCode, nil
}

func CheckStatusPrepareAnnotation(annotation t.CvatAnnotation) (err error) {
	log.Println("cvat.CheckStatusPrepareAnnotation.Started", err)
	defer log.Println("cvat.CheckStatusPrepareAnnotation.Finished", err)
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, AnnotationUrl)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.CheckStatusPrepareAnnotation.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	request.URL.RawQuery = q.Encode()
	for {
		time.Sleep(2 * time.Second)
		response, err := client.Do(request)
		if err != nil {
			log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.CheckStatusPrepareAnnotation.client.Do(request)", err)
			return err
		}
		log.Println(response.StatusCode)
		if response.StatusCode == 201 || response.StatusCode == 202 {
			return nil
		}
		if response.StatusCode == 404 {
			log.Printf("url=%s response.StatusCode=%d", url, response.StatusCode)
			err = fmt.Errorf("url=%s response.StatusCode=%d", url, response.StatusCode)
			return err
		}
	}
}

func CheckStatusPrepareDataset(annotation t.CvatAnnotation) (err error) {
	log.Println("cvat.CheckStatusPrepareDataset.Started", err)
	defer log.Println("cvat.CheckStatusPrepareDataset.Finished", err)
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, DatasetUrl)
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.CheckStatusPrepareDataset.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	request.URL.RawQuery = q.Encode()
	for {
		time.Sleep(2 * time.Second)
		response, err := client.Do(request)
		if err != nil {
			log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.CheckStatusPrepareDataset.client.Do(request)", err)
			return err
		}
		if response.StatusCode == 201 || response.StatusCode == 202 {
			return nil
		}
		if response.StatusCode == 404 {
			log.Printf("url=%s response.StatusCode=%d", url, response.StatusCode)
			err = fmt.Errorf("url=%s response.StatusCode=%d", url, response.StatusCode)
			return err
		}
	}
}

func DownloadAnnotationZip(annotation t.CvatAnnotation, dst string) (err error) {
	log.Println("cvat.DownloadAnnotationZip.Started", err)
	defer log.Println("cvat.DownloadAnnotationZip.Finished", err)
	if err := os.MkdirAll(fp.Dir(dst), 0777); err != nil {
		log.Println("cvat.DownloadAnnotationZip.os.MkdirAll(fp.Dir(dst), 0777)", err)
	}
	out, err := os.Create(dst)
	if err != nil {
		log.Println("Create", err)
		return err
	}
	defer out.Close()
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, AnnotationUrl)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadAnnotationZip.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	q.Add("action", "download")
	request.URL.RawQuery = q.Encode()
	var response *http.Response
	for {

		response, err = client.Do(request)
		if err != nil {
			log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadAnnotationZip.client.Do(request)")
			return err
		}
		if response.StatusCode == 202 {
			response.Body.Close()
			continue
		}
		if response.StatusCode == 200 {
			break
		}
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadAnnotationZip.client.Do(request)", response.StatusCode)
		return nil
	}
	defer response.Body.Close()

	_, err = io.Copy(out, response.Body)
	if err != nil {
		log.Println("Copy", err)
		return err
	}
	return nil
}

func DownloadDatasetZip(annotation t.CvatAnnotation, dst string) (err error) {
	log.Println("cvat.DownloadDatasetZip.Started")
	defer log.Println("cvat.DownloadDatasetZip.Finished")
	if err := os.MkdirAll(fp.Dir(dst), 0777); err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadDatasetZip.os.MkdirAll(fp.Dir(dst), 0777)", err)
	}
	out, err := os.Create(dst)
	if err != nil {
		log.Println("Create", err)
		return err
	}
	defer out.Close()
	client := http.Client{Timeout: 30 * time.Second}
	url := fmt.Sprintf("%s%s%s/%d%s", Host, BaseUrl, TaskUrl, annotation.Id, DatasetUrl)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadDatasetZip.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(User, Password)
	request.Header.Add("Content-Type", "application/json")
	q := request.URL.Query()
	q.Add("format", "COCO 1.0")
	q.Add("action", "download")
	request.URL.RawQuery = q.Encode()
	var response *http.Response
	for {

		response, err = client.Do(request)
		if err != nil {
			log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadDatasetZip.client.Do(request)")
			return err
		}
		if response.StatusCode == 202 {
			response.Body.Close()
			continue
		}
		if response.StatusCode == 200 {
			break
		}
		log.Println("domains.cvat_task.pkg.third_part_api.cvat.cvat.DownloadDatasetZip.client.Do(request)", response.StatusCode)
		return nil
	}
	defer response.Body.Close()

	_, err = io.Copy(out, response.Body)
	if err != nil {
		log.Println("Copy", err)
		return err
	}
	return nil
}
