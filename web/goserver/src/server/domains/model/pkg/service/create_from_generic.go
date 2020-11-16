package service

import (
	"context"
	"fmt"
	"log"
	"os"
	fp "path/filepath"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	buildFindOne "server/db/pkg/handler/build/find_one"
	buildInsertOne "server/db/pkg/handler/build/insert_one"
	modelFindOne "server/db/pkg/handler/model/find_one"
	modelInsertOne "server/db/pkg/handler/model/insert_one"
	modelUpdateOne "server/db/pkg/handler/model/update_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
	statusModelTrain "server/db/pkg/types/status/model/train"
	kitendpoint "server/kit/endpoint"
	u "server/kit/utils"
	"server/kit/utils/basic/arrays"
	ufiles "server/kit/utils/basic/files"
)

type CreateFromGenericRequest struct {
	GenericModelId primitive.ObjectID `json:"genericModelId"`
	ProblemId      primitive.ObjectID `json:"problemId"`
}

func (s *basicModelService) CreateFromGeneric(ctx context.Context, req CreateFromGenericRequest) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)
		genericModel, defaultBuild, problem := s.getGenericModelDefaultBuildProblem(req.GenericModelId, req.ProblemId)
		modelDirPath := createModelDirPath(problem, genericModel.Name)
		modelSnapshotPath := copySnapshot(genericModel.SnapshotPath, modelDirPath)
		model := s.createModelFromGeneric(genericModel, problem, modelDirPath, modelSnapshotPath)
		copyModelFilesFromParentModel(genericModel.Dir, model.Dir, genericModel.TemplatePath, []string{})
		model = s.eval(ctx, model, defaultBuild, problem, false)
		returnChan <- kitendpoint.Response{Data: model, Err: kitendpoint.Error{Code: 0}, IsLast: true}
	}()
	return returnChan
}

func copyModelFilesFromParentModel(from, to, modelTemplatePath string, excluded []string) {
	templateYamlPath := copyTemplateYaml(modelTemplatePath, to)
	templateYaml := getTemplateYaml(templateYamlPath)
	copyConfig(from, to, templateYaml)
	copyDependenciesFromParentModel(from, to, templateYaml, excluded)
	saveMetrics(to, templateYaml)
}

func copyDependenciesFromParentModel(from, to string, modelYml ModelYml, excluded []string) {
	for _, d := range modelYml.Dependencies {
		if arrays.ContainsString(excluded, d.Destination) {
			continue
		}
		fromPath := fp.Join(from, d.Destination)
		toPath := fp.Join(to, d.Destination)
		log.Println(fromPath, toPath)
		if err := copyFiles(fromPath, toPath); err != nil {
			log.Println("create_from_generic.copyDependenciesFromParentModel.copyFiles(fromPath, toPath)", err)
		}
	}
}

func (s *basicModelService) updateModelEvaluateStatus(ctx context.Context, model t.Model, buildId primitive.ObjectID, status string) t.Model {
	if model.Evaluates == nil {
		model.Evaluates = make(map[string]t.Evaluate)
	}
	model.Evaluates[buildId.Hex()] = t.Evaluate{
		Metrics: model.Evaluates[buildId.Hex()].Metrics,
		Status:  status,
	}
	log.Println("updateModelEvaluateStatus", model.Evaluates)
	modelUpdateOneResp := <-modelUpdateOne.Send(ctx, s.Conn, model)
	log.Println("updateModelEvaluateStatus", modelUpdateOneResp.Data.(modelUpdateOne.ResponseData).Evaluates)
	return modelUpdateOneResp.Data.(modelUpdateOne.ResponseData)
}

func (s *basicModelService) createModelFromGeneric(genericModel t.Model, problem t.Problem, dir, snapshotPath string) t.Model {
	modelInsertOneResp := <-modelInsertOne.Send(context.TODO(), s.Conn, modelInsertOne.RequestData{
		ConfigPath:    fp.Join(dir, "model.py"),
		Dir:           dir,
		Evaluates:     make(map[string]t.Evaluate),
		ProblemId:     problem.Id,
		Name:          genericModel.Name,
		ParentModelId: genericModel.Id,
		Scripts: t.Scripts{
			Train: fp.Join(dir, "train.py"),
			Eval:  fp.Join(dir, "eval.py"),
		},
		SnapshotPath:   snapshotPath,
		Status:         statusModelTrain.Default,
		TemplatePath:   fp.Join(dir, "template.yaml"),
		TrainingGpuNum: genericModel.TrainingGpuNum,
	})

	return modelInsertOneResp.Data.(modelInsertOne.ResponseData)
}

func getClasses(labels []map[string]interface{}) string {
	var classesArr []string
	for _, label := range labels {
		classesArr = append(classesArr, fmt.Sprintf("%s", label["name"].(string)))
	}
	return strings.Join(classesArr, ",")
}

func (s *basicModelService) getGenericModelDefaultBuildProblem(genericModelId, problemId primitive.ObjectID) (t.Model, t.Build, t.Problem) {
	problemFindOneChan := problemFindOne.Send(context.TODO(), s.Conn, problemFindOne.RequestData{
		Id: problemId,
	})
	modelFindOneChan := modelFindOne.Send(context.TODO(), s.Conn, modelFindOne.RequestData{
		Id: genericModelId,
	})
	buildFindOneChan := buildFindOne.Send(context.TODO(), s.Conn, buildFindOne.RequestData{
		ProblemId: problemId,
		Name:      "default",
	})
	problemFindOneResp, modelFindOneResp, buildFindOneResp := <-problemFindOneChan, <-modelFindOneChan, <-buildFindOneChan
	problem := problemFindOneResp.Data.(problemFindOne.ResponseData)
	model := modelFindOneResp.Data.(modelFindOne.ResponseData)
	build := buildFindOneResp.Data.(buildFindOne.ResponseData)
	if build.Id.IsZero() {
		buildInsertOneResp := <-buildInsertOne.Send(context.TODO(), s.Conn, buildInsertOne.RequestData{
			ProblemId: problemId,
			Folder:    "_default",
			Name:      "default",
			Status:    buildStatus.Ready,
		})
		build = buildInsertOneResp.Data.(buildInsertOne.ResponseData)
	}
	return model, build, problem
}

func createModelDirPath(problem t.Problem, genericModelName string) string {
	classFolderName := u.StringToFolderName(problem.Class)
	titleFolderName := u.StringToFolderName(problem.Title)
	path := fp.Join("/problem", classFolderName, titleFolderName, genericModelName)
	if err := os.MkdirAll(path, 0777); err != nil {
		log.Println("domains.problem.pkg.service.create.createModelDirPath.os.MkdirAll(path, 0777)", err)
	}
	return path
}

func copySnapshot(genericSnapshotPath, modelDirPath string) string {
	snapshotPath := fp.Join(modelDirPath, fp.Base(genericSnapshotPath))
	if _, err := ufiles.Copy(genericSnapshotPath, snapshotPath); err != nil {
		log.Println("domains.problem.pkg.service.create.copySnapshot.ufiles.Copy(genericSnapshotPath, snapshotPath)")
	}
	return snapshotPath
}
