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
	modelFindOne "server/db/pkg/handler/model/find_one"
	modelUpdateOne "server/db/pkg/handler/model/update_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	problemType "server/db/pkg/types/problem/types"
	statusModelEvaluate "server/db/pkg/types/status/model/evaluate"
	kitendpoint "server/kit/endpoint"
	"server/kit/utils/basic/arrays"
)

type EvaluateRequest struct {
	ModelId   primitive.ObjectID `json:"modelId" bson:"modelId"`
	BuildId   primitive.ObjectID `json:"buildId" bson:"buildId"`
	ProblemId primitive.ObjectID `json:"problemId" bson:"problemId"`
}

func (s *basicModelService) Evaluate(ctx context.Context, req EvaluateRequest) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)
		model, build, problem := s.getModelBuildProblem(req.ModelId, req.BuildId, req.ProblemId)
		model = s.eval(ctx, model, build, problem, false)
		returnChan <- kitendpoint.Response{Data: model, Err: kitendpoint.Error{Code: 0}, IsLast: true}
	}()
	return returnChan
}

func (s *basicModelService) eval(ctx context.Context, model t.Model, build t.Build, problem t.Problem, saveImages bool) t.Model {
	model = s.updateModelEvaluateStatus(ctx, model, build.Id, statusModelEvaluate.InProgress)
	evalFolderPath := createEvalDir(model.Dir, build.Folder)
	metricsYml := fp.Join(evalFolderPath, "metrics.yaml")
	outputImagesPath := ""
	if saveImages == true {
		outputImagesPath = makeImagesFolder(evalFolderPath)
	}
	commands := s.prepareEvaluateCommands(metricsYml, outputImagesPath, model, build, problem)
	outputLog := createFile(fp.Join(evalFolderPath, "output.log"))
	env := getEvaluateEnv()
	if err := s.runCommand(commands, env, model.Dir, outputLog); err != nil {
		model = s.updateModelEvaluateStatus(ctx, model, build.Id, statusModelEvaluate.Failed)
	} else {
		model = s.saveModelEvalMetrics(metricsYml, build.Id, model)
	}
	log.Println("eval.model.Evaluates", model.Evaluates)
	return model
}

func makeImagesFolder(evalFolderPath string) string {
	outputImagesPath := fp.Join(evalFolderPath, "output_images")
	if err := os.MkdirAll(outputImagesPath, 0777); err != nil {
		log.Println("evaluate.createFolder.os.MkdirAll(path, 0777)", err)
	}
	return outputImagesPath
}

func getEvaluateEnv() []string {
	return []string{
		"MMDETECTION_DIR=/ote/external/mmdetection",
	}
}

func createFile(name string) string {
	_, err := os.Create(name)
	if err != nil {
		log.Println("evaluate.createMetricsYaml.os.Create(metricsYml)", err)
	}
	return name
}

func (s *basicModelService) getModelBuildProblem(modelId, buildId, problemId primitive.ObjectID) (t.Model, t.Build, t.Problem) {
	chModel := modelFindOne.Send(
		context.TODO(),
		s.Conn,
		modelFindOne.RequestData{Id: modelId},
	)
	chBuild := buildFindOne.Send(
		context.TODO(),
		s.Conn,
		buildFindOne.RequestData{Id: buildId},
	)
	chProblem := problemFindOne.Send(
		context.TODO(),
		s.Conn,
		problemFindOne.RequestData{Id: problemId},
	)
	rModel, rBuild, rProblem := <-chModel, <-chBuild, <-chProblem
	model := rModel.Data.(modelFindOne.ResponseData)
	build := rBuild.Data.(buildFindOne.ResponseData)
	problem := rProblem.Data.(problemFindOne.ResponseData)

	return model, build, problem
}

func createEvalDir(modelDirPath, buildName string) string {
	newModelDirPath := fmt.Sprintf("%s/%s", modelDirPath, buildName)
	if err := os.Mkdir(newModelDirPath, 0777); err != nil {
		log.Println("domains.model.pkg.service.evaluate.createEvalDir.os.Mkdir(newModelDirPath, 0777)", err)
	}
	return newModelDirPath
}

func (s *basicModelService) prepareEvaluateCommands(evalYml, outputImagesPath string, model t.Model, build t.Build, problem t.Problem) []string {
	evalDir := fp.Dir(evalYml)
	if err := os.MkdirAll(evalDir, 0777); err != nil {
		log.Println("domains.model.pkg.service.evaluate.prepareEvaluateCommands.os.MkdirAll(evalFolder, 0777)", err)
	}
	if err := os.Chmod(model.Scripts.Eval, 0777); err != nil {
		log.Println("domains.model.pkg.service.evaluate.prepareEvaluateCommands.os.Chmod(script, 0777)", err)
	}
	imgPrefix, annFile := s.getImgPrefixAndAnnotation("test", build, problem)
	imgPrefixStr := strings.Join(imgPrefix, ",")
	annFileStr := strings.Join(annFile, ",")
	paramsArr := []string{
		fmt.Sprintf("--load-weights %s", model.SnapshotPath),
		fmt.Sprintf("--save-metrics-to %s", evalYml),
		fmt.Sprintf("--test-ann-files %s", annFileStr),
		fmt.Sprintf("--test-data-roots %s", imgPrefixStr),
	}

	if arrays.ContainsString([]string{problemType.Custom, problemType.Generic}, problem.Type) {
		classes := getClasses(problem.Labels)
		paramsArr = append(paramsArr, fmt.Sprintf("--classes %s", classes))
	}
	if outputImagesPath != "" {
		paramsArr = append(paramsArr, fmt.Sprintf("--save-output-to %s", outputImagesPath))
	}
	paramsStr := strings.Join(paramsArr, " ")
	commands := []string{
		fmt.Sprintf(`pip install -r %s`, fp.Join(model.Dir, "requirements.txt")),
		fmt.Sprintf(`python %s %s`, model.Scripts.Eval, paramsStr),
	}
	return commands
}

func (s *basicModelService) saveModelEvalMetrics(evalYml string, buildId primitive.ObjectID, model t.Model) t.Model {
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
	if model.Evaluates == nil {
		model.Evaluates = make(map[string]t.Evaluate)
	}
	model.Evaluates[buildId.Hex()] = t.Evaluate{
		Metrics: metrics.Metrics,
		Status:  statusModelEvaluate.Finished,
	}
	modelUpdateOneResp := <-modelUpdateOne.Send(context.TODO(), s.Conn, model)
	return modelUpdateOneResp.Data.(modelUpdateOne.ResponseData)
}
