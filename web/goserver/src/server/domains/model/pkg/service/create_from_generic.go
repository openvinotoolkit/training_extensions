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
	modelStatus "server/db/pkg/types/status/model"
	kitendpoint "server/kit/endpoint"
	u "server/kit/utils"
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
		copyTemplateYamlToNewModel(genericModel, model)
		copyModelPy(genericModel.ConfigPath, model.ConfigPath)
		model = s.initEval(model, defaultBuild, problem)
		model = s.saveModel(model)
		returnChan <- kitendpoint.Response{Data: model, Err: nil, IsLast: true}
	}()
	return returnChan
}

func (s *basicModelService) initEval(model t.Model, build t.Build, problem t.Problem) t.Model {
	evalFolderPath := createEvalDir(model.Dir, build.Folder)
	metricsYml := fp.Join(evalFolderPath, "metrics.yaml")
	commands := s.prepareEvaluateCommands(metricsYml, "", model, build, problem)
	outputLog := createFile(fp.Join(evalFolderPath, "output.log"))
	env := getEvaluateEnv(model)
	s.runCommand(commands, env, problem.ToolsPath, outputLog)
	model = s.saveModelEvalMetrics(metricsYml, build.Id, model)
	return model
}

func copyModelPy(from, to string) {
	log.Println("copyModelPy", from, to)
	if _, err := ufiles.Copy(from, to); err != nil {
		log.Println("domains.model.pkg.service.create_from_generic.copyModelPy.ufiles.Copy(from, to)", err)
	}
}

func (s *basicModelService) createModelFromGeneric(genericModel t.Model, problem t.Problem, dir, snapshotPath string) t.Model {
	modelInsertOneResp := <-modelInsertOne.Send(context.TODO(), s.Conn, modelInsertOne.RequestData{
		ConfigPath:    fp.Join(dir, "model.py"),
		Dir:           dir,
		ProblemId:     problem.Id,
		Metrics:       make(map[string][]t.Metric),
		Name:          genericModel.Name,
		ParentModelId: genericModel.Id,
		Scripts: t.Scripts{
			Train: fp.Join(problem.ToolsPath, "train.py"),
			Eval:  fp.Join(problem.ToolsPath, "eval.py"),
		},
		SnapshotPath:      snapshotPath,
		Status:            modelStatus.Initiate,
		TemplatePath:      fp.Join(dir, "template.yaml"),
		TensorBoardLogDir: "",
		TrainingGpuNum:    genericModel.TrainingGpuNum,
		TrainingWorkDir:   "",
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
	log.Println("BEFORE", build)
	if build.Id.IsZero() {
		buildInsertOneResp := <-buildInsertOne.Send(context.TODO(), s.Conn, buildInsertOne.RequestData{
			ProblemId: problemId,
			Folder:    "_default",
			Name:      "default",
			Status:    buildStatus.Ready,
		})
		build = buildInsertOneResp.Data.(buildInsertOne.ResponseData)
	}
	log.Println("AFTER", build)
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

func (s *basicModelService) saveModel(model t.Model) t.Model {
	model.Status = modelStatus.Default
	modelUpdateOneResp := <-modelUpdateOne.Send(context.TODO(), s.Conn, model)
	return modelUpdateOneResp.Data.(modelUpdateOne.ResponseData)
}
