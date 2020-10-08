package service

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	"gopkg.in/yaml.v2"

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
		modelDirPath := createModelDirPath(problem.Dir, genericModel.Name)
		modelSnapshotPath := copySnapshot(genericModel.SnapshotPath, modelDirPath)
		model := s.createModelFromGeneric(genericModel, problem.Id, modelDirPath, modelSnapshotPath)
		metricYml := s.initialModelEval(model, problem)
		copyConfigPath(genericModel.ConfigPath, &model)
		saveMetricsToModel(metricYml, defaultBuild.Id, &model)
		model = s.saveModel(model)
		returnChan <- kitendpoint.Response{Data: model, Err: nil, IsLast: true}
	}()
	return returnChan
}

func copyConfigPath(genericModelConfigPath string, model *t.Model) {
	modelConfigPath := fp.Join(model.Dir, fp.Base(genericModelConfigPath))
	if _, err := ufiles.Copy(genericModelConfigPath, modelConfigPath); err != nil {
		log.Println("domains.model.pkg.service.create_from_generic.copyConfigPath.ufiles.Copy(genericModelConfigPath, modelConfigPath)")
	}
	model.ConfigPath = modelConfigPath
}

func (s *basicModelService) createModelFromGeneric(genericModel t.Model, problemId primitive.ObjectID, dir, snapshotPath string) t.Model {
	modelInsertOneResp := <-modelInsertOne.Send(context.TODO(), s.Conn, modelInsertOne.RequestData{
		ConfigPath:        genericModel.ConfigPath,
		Dir:               dir,
		ProblemId:         problemId,
		Metrics:           make(map[string][]t.Metric),
		Name:              genericModel.Name,
		ParentModelId:     genericModel.Id,
		SnapshotPath:      snapshotPath,
		Status:            modelStatus.InProgress,
		TensorBoardLogDir: "",
		TrainingGpuNum:    genericModel.TrainingGpuNum,
		TrainingWorkDir:   "",
	})

	return modelInsertOneResp.Data.(modelInsertOne.ResponseData)
}

func (s *basicModelService) initialModelEval(model t.Model, problem t.Problem) string {
	evalYml := fmt.Sprintf("%s/model.yml", model.Dir)
	commands := s.prepareInitialEvaluateCommands(model.Scripts.Eval, evalYml, model, problem)
	outputLog := fmt.Sprintf("%s/output.log", model.Dir)
	s.runCommand(commands, []string{}, model.TrainingWorkDir, outputLog)
	return evalYml
}

func (s *basicModelService) prepareInitialEvaluateCommands(script, evalYml string, model t.Model, problem t.Problem) []string {
	if err := os.MkdirAll(fp.Dir(evalYml), 0777); err != nil {
		log.Println("domains.model.pkg.service.create_from_generic.prepareEvaluateCommands.os.MkdirAll(evalFolder, 0777)", err)
		return []string{}
	}
	if err := os.Chmod(script, 0777); err != nil {
		log.Println("domains.model.pkg.service.create_from_generic.prepareEvaluateCommands.os.Chmod(script, 0777)", err)
		return []string{}
	}
	classes, numClasses := getClasses(problem.Labels)
	configArr := []string{
		"data.test.ann_file=None",
		"data.test.img_prefix=None",
		fmt.Sprintf("data.test.classes=(%s,)", classes),
		"data.train.ann_file=None",
		"data.train.img_prefix=None",
		fmt.Sprintf("data.train.classes=(%s,)", classes),
		"data.val.ann_file=None",
		"data.val.img_prefix=None",
		fmt.Sprintf("data.val.classes=(%s,)", classes),
		fmt.Sprintf("model.bbox_head.num_classes=%d", numClasses),
	}
	configUpdatedFields := strings.Join(configArr, " ")
	commands := []string{
		fmt.Sprintf(`python %s %s %s %s --update_config %q`, script, model.ConfigPath, model.SnapshotPath, evalYml, configUpdatedFields),
	}
	return commands
}

func getClasses(labels []map[string]interface{}) (string, int) {
	var classesArr []string
	for _, label := range labels {
		classesArr = append(classesArr, fmt.Sprintf("'%s'", label["name"].(string)))
	}
	return strings.Join(classesArr, ","), len(classesArr)
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
			Name:      "default",
			Status:    buildStatus.Ready,
		})
		build = buildInsertOneResp.Data.(buildInsertOne.ResponseData)
	}
	log.Println("AFTER", build)
	return model, build, problem
}

func createModelDirPath(genericDirPath, genericModelName string) string {
	path := fp.Join("/problem", genericDirPath, "models", genericModelName)
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

func saveMetricsToModel(evalYml string, buildId primitive.ObjectID, model *t.Model) {
	log.Println(evalYml)
	newModelYamlFile, err := ioutil.ReadFile(evalYml)
	if err != nil {
		log.Println("ReadFile", err)
	}
	var metrics struct {
		Metrics []t.Metric `yaml:"metrics"`
	}
	err = yaml.Unmarshal(newModelYamlFile, &metrics)
	if err != nil {
		log.Println("Unmarshal", err)
	}
	model.Metrics = make(map[string][]t.Metric)
	model.Metrics[buildId.Hex()] = metrics.Metrics
}

func (s *basicModelService) saveModel(model t.Model) t.Model {
	model.Status = modelStatus.Finished
	modelUpdateOneResp := <-modelUpdateOne.Send(context.TODO(), s.Conn, model)
	return modelUpdateOneResp.Data.(modelUpdateOne.ResponseData)
}
