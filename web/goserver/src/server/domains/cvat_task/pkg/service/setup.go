package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"

	assetFind "server/db/pkg/handler/asset/find"
	assetFindOne "server/db/pkg/handler/asset/find_one"
	buildFindOne "server/db/pkg/handler/build/find_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	cvatTaskFindOne "server/db/pkg/handler/cvat_task/find_one"
	buildStatus "server/db/pkg/types/build/status"
	typeAsset "server/db/pkg/types/type/asset"
	kitendpoint "server/kit/endpoint"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"go.mongodb.org/mongo-driver/bson/primitive"

	cvatTaskUpdateOne "server/db/pkg/handler/cvat_task/update_one"
	t "server/db/pkg/types"
)

const (
	cvatHost     = "http://cvat:8080"
	cvatBaseUrl  = "/api/v1"
	cvatTaskUrl  = "/tasks"
	cvatUser     = "django"
	cvatPassword = "django"
)

type SetupRequestData struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

type CvatCreateTaskResponseBody = t.CvatAnnotation

func (s *basicCvatTaskService) Setup(ctx context.Context, req SetupRequestData) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)

		cvatTaskFindOneResp := <-cvatTaskFindOne.Send(
			ctx,
			s.Conn,
			cvatTaskFindOne.RequestData{
				Id: req.Id,
			},
		)
		cvatTask := cvatTaskFindOneResp.Data.(cvatTaskFindOne.ResponseData)
		assetFindOneResp := <-assetFindOne.Send(
			ctx,
			s.Conn,
			assetFindOne.RequestData{
				Id: cvatTask.AssetId,
			},
		)
		asset := assetFindOneResp.Data.(assetFindOne.ResponseData)
		if asset.Type == typeAsset.Folder {
			assetFindResp := <-assetFind.Send(
				ctx,
				s.Conn,
				assetFind.RequestData{
					ParentFolder: strings.Join([]string{asset.ParentFolder, asset.Name}, "/"),
				},
			)
			assets := assetFindResp.Data.(assetFind.ResponseData).Items
			var assetIds []primitive.ObjectID
			for _, asset := range assets {
				assetIds = append(assetIds, asset.Id)
			}
			cvatTaskFindResp := <-cvatTaskFind.Send(
				ctx,
				s.Conn,
				cvatTaskFind.RequestData{
					AssetIds:  assetIds,
					ProblemId: cvatTask.ProblemId,
				},
			)
			childrenCvatTasks := cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
			for _, childCvatTask := range childrenCvatTasks {
				s.Setup(ctx, SetupRequestData{
					Id: childCvatTask.Id,
				})
			}
			returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: kitendpoint.Error{Code: 0}}
		}

		if isCvatTaskExists(cvatTask) {
			return
		}
		cvatTask.Status = "pushInProgress"
		cvatTaskUpdateOne.Send(context.TODO(), s.Conn, cvatTask)
		buildFindOneResp := <-buildFindOne.Send(
			ctx,
			s.Conn,
			buildFindOne.RequestData{
				ProblemId: cvatTask.ProblemId,
				Status:    buildStatus.Tmp,
			},
		)
		tmpBuild := buildFindOneResp.Data.(buildFindOne.ResponseData)
		buildSplit := findBuildAssetSplit(tmpBuild, asset)
		returnChan <- kitendpoint.Response{IsLast: false, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 0}}
		cvatCreateTaskResponseBody := createCvatTask(cvatTask)
		cvatTask.Annotation = cvatCreateTaskResponseBody

		cvatTaskUpdateOneResp := <-cvatTaskUpdateOne.Send(
			ctx,
			s.Conn,
			cvatTask,
		)
		cvatTask = cvatTaskUpdateOneResp.Data.(cvatTaskUpdateOne.ResponseData)

		err := addDataToCvatTask(cvatTask, cvatTask.Annotation.Id)
		if err != nil {
			log.Fatalln("domains.cvat_task.pkg.service.setup_one.SetupOne.addDataToCvatTask(cvatTask)", err)
		}

		cvatTask, err = monitorCreateTaskStatus(ctx, s.Conn, cvatTask)
		if err != nil {
			deleteTaskFromCvat(cvatTask)
			returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: kitendpoint.Error{Code: 1}}
		}

		cvatTask, err = getCvatTask(cvatTask)
		if err != nil {
			returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: kitendpoint.Error{Code: 1}}
		}
		cvatTask.Status = "initialPullReady"
		cvatTaskUpdateOneResp = <-cvatTaskUpdateOne.Send(
			ctx,
			s.Conn,
			cvatTask,
		)
		cvatTask = cvatTaskUpdateOneResp.Data.(cvatTaskUpdateOne.ResponseData)
		buildFindOneResp = <-buildFindOne.Send(
			ctx,
			s.Conn,
			buildFindOne.RequestData{
				ProblemId: cvatTask.ProblemId,
				Status:    buildStatus.Tmp,
			},
		)
		tmpBuild = buildFindOneResp.Data.(buildFindOne.ResponseData)
		buildSplit = findBuildAssetSplit(tmpBuild, asset)

		returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 0}}
	}()
	return returnChan

}

func isCvatTaskExists(cvatTask t.CvatTask) bool {
	if cvatTask.Annotation.Status != "" {
		return true
	}
	return false
}

func createCvatTask(cvatTask t.CvatTask) (responseBody CvatCreateTaskResponseBody) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s", cvatHost, cvatBaseUrl, cvatTaskUrl)

	body, err := json.Marshal(cvatTask.Params)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.json.Marshal(cvatTask.Params)", err)
	}

	log.Println(string(body))

	request, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.jhttp.NewRequest(\"POST\",url, bytes.NewBuffer(body))", err)
	}

	request.SetBasicAuth(cvatUser, cvatPassword)
	request.Header.Add("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.client.Do", err)
	}
	defer response.Body.Close()

	if response.StatusCode != 201 {
		log.Println("domains.cvat_task.pkg.service.setup_one.createCvatTask.response.StatusCode=", response.StatusCode, "Expected=", 201)
		return
	}

	responseBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.ioutil.ReadAll(response.Body)", err)
	}

	err = json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.json.Unmarshal(responseBytes, &cvatCreateTaskResponseBody)", err)
	}
	responseBody.Url = "/" + strings.Join(strings.Split(responseBody.Url, "/")[3:], "/")
	return responseBody
}

func addDataToCvatTask(cvatTask t.CvatTask, cvatCreateTaskResponseBodyId int) (err error) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s/%d/data", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatCreateTaskResponseBodyId)
	body, err := json.Marshal(map[string]interface{}{
		"server_files":   []string{cvatTask.AssetPath},
		"image_quality":  70,
		"use_zip_chunks": true,
	})

	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.json.Marshal(map[string][]string{", err)
		return
	}
	request, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	log.Println("Send POST request", url)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.http.NewRequest(\"POST\", url, bytes.NewBuffer(body))", err)
		return
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	request.Header.Add("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.ent.Do(request)", err)
		return
	}
	if response.StatusCode != 202 {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.response.StatusCode=", response.StatusCode, "Expected=", 202)
		return fmt.Errorf("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.response.StatusCode=%d, Expected=%d", response.StatusCode, 202)
	}
	return
}

func monitorCreateTaskStatus(ctx context.Context, conn *rabbitmq.Connection, cvatTask t.CvatTask) (t.CvatTask, error) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s/%d/status", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatTask.Annotation.Id)

	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.http.NewRequest(\"GET\", url, nil)", err)
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	request.Header.Add("Content-Type", "application/json")
	for {
		time.Sleep(2 * time.Second)
		response, err := client.Do(request)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.client.Do(request)", err)
		}
		if response.StatusCode != 200 {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.response.StatusCode=", response.StatusCode, "Expected=", 200)
		}
		statusResponseBytes, err := ioutil.ReadAll(response.Body)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne. ioutil.ReadAll(response.Body)", err)
		}
		response.Body.Close()
		var responseBody t.CvatTaskCreateTaskStatus
		err = json.Unmarshal(statusResponseBytes, &responseBody)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.json.Unmarshal(statusResponseBytes, &responseBody)", err)
		}
		cvatTask.CreateTaskStatus = responseBody
		updateOneResp := <-cvatTaskUpdateOne.Send(
			ctx,
			conn,
			cvatTask,
		)
		cvatTask = updateOneResp.Data.(cvatTaskUpdateOne.ResponseData)
		if responseBody.State == "Finished" {
			return cvatTask, nil
		}
		if responseBody.State == "Failed" {
			return cvatTask, fmt.Errorf("responseBody.State=\"Failed\"")
		}
	}
}

func deleteTaskFromCvat(cvatTask t.CvatTask) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s/%d", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatTask.Annotation.Url)
	request, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.deleteTaskFromCvat.http.NewRequest(\"DELETE\",", url, ", nil)", err)
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	_, err = client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.ent.Do(request)", err)
		return
	}
}

type CvatTaskJob struct {
	Url    string `json:"url"`
	Id     int64  `json:"id"`
	Status string `json:"status"`
}

type CvatTaskSegment struct {
	StartFrame int64         `bson:"start_frame" json:"start_frame"`
	StopFrame  int64         `bson:"stop_frame" json:"stop_frame"`
	Jobs       []CvatTaskJob `bson:"jobs" json:"jobs"`
}

type GetCvatTaskResponse struct {
	Segments []CvatTaskSegment `json:"segments"`
}

func getCvatTask(cvatTask t.CvatTask) (t.CvatTask, error) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s/%d", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatTask.Annotation.Id)
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getCvatTask.http.NewRequest(\"GET\",", url, ", nil)", err)
		return cvatTask, err
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getCvatTask.ent.Do(request)", err)
		return cvatTask, err
	}
	responseBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getCvatTask.ioutil.ReadAll(response.Body)", err)
		return cvatTask, err
	}
	response.Body.Close()
	var responseBody GetCvatTaskResponse
	err = json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getCvatTask.json.Unmarshal(statusResponseBytes, &responseBody)", err)
		return cvatTask, err
	}
	var jobs []t.CvatJob
	for _, segment := range responseBody.Segments {
		for _, job := range segment.Jobs {
			jobs = append(jobs, t.CvatJob{
				StartFrame: segment.StartFrame,
				StopFrame:  segment.StopFrame,
				Url:        job.Url,
				Id:         job.Id,
				Status:     job.Status,
			})
		}

	}
	cvatTask.Annotation.Jobs = jobs
	return cvatTask, nil
}
