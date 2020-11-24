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
	statusCvatTask "server/db/pkg/types/status/cvatTask"
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
		asset, cvatTask := s.getAssetCvatTask(ctx, req.Id)
		originalCvatTask := cvatTask

		if asset.Type == typeAsset.Folder {
			assetIds := s.getChildrenAssetIds(ctx, asset)
			childrenCvatTasks := s.getCvatTaskByAssets(ctx, assetIds, cvatTask.ProblemId)
			for _, childCvatTask := range childrenCvatTasks {
				s.Setup(ctx, SetupRequestData{Id: childCvatTask.Id})
			}
			returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: kitendpoint.Error{Code: 0}}
		}

		if isCvatTaskExists(cvatTask) {
			return
		}
		cvatTask = s.updateCvatTaskStatus(ctx, cvatTask, statusCvatTask.PushInProgress)
		tmpBuild := s.getTmpBuild(ctx, cvatTask.ProblemId)
		buildSplit := findBuildAssetSplit(tmpBuild, asset)
		returnChan <- kitendpoint.Response{IsLast: false, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 0}}
		cvatTask.Annotation = createCvatTask(cvatTask)
		cvatTask = s.updateCvatTask(ctx, cvatTask)
		cvatTask, err := addDataToCvatTask(cvatTask, cvatTask.Annotation.Id)
		if err != nil {
			cvatTask = s.updateCvatTask(ctx, originalCvatTask)
			returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 1}}
			log.Fatalln("domains.cvat_task.pkg.service.setup_one.SetupOne.addDataToCvatTask(cvatTask)", err)
		}

		cvatTask, err = monitorCreateTaskStatus(ctx, s.Conn, cvatTask)
		if err != nil {
			deleteTaskFromCvat(cvatTask)
			cvatTask = s.updateCvatTask(ctx, originalCvatTask)
			returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 1}}
			return
		}

		cvatTask, err = getTaskFromCvat(cvatTask)
		if err != nil {
			cvatTask = s.updateCvatTask(ctx, originalCvatTask)
			returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 1}}
		}
		cvatTask = s.updateCvatTaskStatus(ctx, cvatTask, statusCvatTask.InitialPullReady)
		tmpBuild = s.getTmpBuild(ctx, cvatTask.ProblemId)
		buildSplit = findBuildAssetSplit(tmpBuild, asset)
		returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: kitendpoint.Error{Code: 0}}
	}()
	return returnChan

}

func (s *basicCvatTaskService) getAssetCvatTask(ctx context.Context, id primitive.ObjectID) (t.Asset, t.CvatTask) {
	cvatTaskFindOneResp := <-cvatTaskFindOne.Send(
		ctx,
		s.Conn,
		cvatTaskFindOne.RequestData{
			Id: id,
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
	return asset, cvatTask
}

func (s *basicCvatTaskService) getChildrenAssetIds(ctx context.Context, parentAsset t.Asset) []primitive.ObjectID {
	childrenAssetFindResp := <-assetFind.Send(
		ctx,
		s.Conn,
		assetFind.RequestData{
			ParentFolder: strings.Join([]string{parentAsset.ParentFolder, parentAsset.Name}, "/"),
		},
	)
	childrenAssets := childrenAssetFindResp.Data.(assetFind.ResponseData).Items
	var assetIds []primitive.ObjectID
	for _, asset := range childrenAssets {
		assetIds = append(assetIds, asset.Id)
	}
	return assetIds
}

func (s *basicCvatTaskService) getCvatTaskByAssets(ctx context.Context, assetIds []primitive.ObjectID, problemId primitive.ObjectID) []t.CvatTask {
	cvatTaskFindResp := <-cvatTaskFind.Send(
		ctx,
		s.Conn,
		cvatTaskFind.RequestData{
			AssetIds:  assetIds,
			ProblemId: problemId,
		},
	)
	return cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
}

func (s *basicCvatTaskService) updateCvatTaskStatus(ctx context.Context, cvatTask t.CvatTask, status string) t.CvatTask {
	cvatTask.Status = status
	cvatTaskUpdateOneResp := <-cvatTaskUpdateOne.Send(ctx, s.Conn, cvatTask)
	return cvatTaskUpdateOneResp.Data.(cvatTaskUpdateOne.ResponseData)
}

func (s *basicCvatTaskService) getTmpBuild(ctx context.Context, problemId primitive.ObjectID) t.Build {
	buildFindOneResp := <-buildFindOne.Send(
		ctx,
		s.Conn,
		buildFindOne.RequestData{
			ProblemId: problemId,
			Status:    buildStatus.Tmp,
		},
	)
	return buildFindOneResp.Data.(buildFindOne.ResponseData)
}

func (s *basicCvatTaskService) updateCvatTask(ctx context.Context, cvatTask t.CvatTask) t.CvatTask {
	cvatTaskUpdateOneResp := <-cvatTaskUpdateOne.Send(
		ctx,
		s.Conn,
		cvatTask,
	)
	return cvatTaskUpdateOneResp.Data.(cvatTaskUpdateOne.ResponseData)
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

func addDataToCvatTask(cvatTask t.CvatTask, cvatCreateTaskResponseBodyId int) (t.CvatTask, error) {
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
		return cvatTask, err
	}
	request, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	log.Println("Send POST request", url)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.http.NewRequest(\"POST\", url, bytes.NewBuffer(body))", err)
		return cvatTask, err
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	request.Header.Add("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.ent.Do(request)", err)
		return cvatTask, err
	}
	if response.StatusCode != 202 {
		log.Println("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.response.StatusCode=", response.StatusCode, "Expected=", 202)
		return cvatTask, fmt.Errorf("domains.cvat_task.pkg.service.setup_one.addDataToCvatTask.response.StatusCode=%d, Expected=%d", response.StatusCode, 202)
	}
	return cvatTask, err
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
			return cvatTask, err
		}
		if response.StatusCode != 200 {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.response.StatusCode=", response.StatusCode, "Expected=", 200)
			return cvatTask, err
		}
		statusResponseBytes, err := ioutil.ReadAll(response.Body)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.ioutil.ReadAll(response.Body)", err)
			return cvatTask, err
		}
		response.Body.Close()
		var responseBody t.CvatTaskCreateTaskStatus
		err = json.Unmarshal(statusResponseBytes, &responseBody)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.setup_one.SetupOne.json.Unmarshal(statusResponseBytes, &responseBody)", err)
			return cvatTask, err
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
	url := fmt.Sprintf("%s%s%s%s", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatTask.Annotation.Url)
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

func getTaskFromCvat(cvatTask t.CvatTask) (t.CvatTask, error) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}
	url := fmt.Sprintf("%s%s%s/%d", cvatHost, cvatBaseUrl, cvatTaskUrl, cvatTask.Annotation.Id)
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getTaskFromCvat.http.NewRequest(\"GET\",", url, ", nil)", err)
		return cvatTask, err
	}
	request.SetBasicAuth(cvatUser, cvatPassword)
	response, err := client.Do(request)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getTaskFromCvat.ent.Do(request)", err)
		return cvatTask, err
	}
	responseBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getTaskFromCvat.ioutil.ReadAll(response.Body)", err)
		return cvatTask, err
	}
	response.Body.Close()
	var responseBody GetCvatTaskResponse
	err = json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		log.Println("domains.cvat_task.pkg.service.setup_one.getTaskFromCvat.json.Unmarshal(statusResponseBytes, &responseBody)", err)
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
