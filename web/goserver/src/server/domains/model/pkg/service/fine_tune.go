package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"
	"regexp"
	"strconv"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	assetFindOne "server/db/pkg/handler/asset/find_one"
	buildFindOne "server/db/pkg/handler/build/find_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	modelFindOne "server/db/pkg/handler/model/find_one"
	modelInsertOne "server/db/pkg/handler/model/insert_one"
	modelUpdateOne "server/db/pkg/handler/model/update_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	splitState "server/db/pkg/types/build/split_state"
	modelStatus "server/db/pkg/types/status/model"
	kitendpoint "server/kit/endpoint"
	"server/kit/utils/basic/arrays"
	ufiles "server/kit/utils/basic/files"
	trainingWorkerGpuNum "server/workers/train/pkg/handler/get_gpu_amount"
	runCommandsWorker "server/workers/train/pkg/handler/run_commands"
)

type FineTuneRequestData struct {
	BatchSize              int    `json:"batchSize"`
	BuildId                string `json:"buildId"`
	Epochs                 int    `json:"epochs"`
	GpuNum                 int    `json:"gpuNumber"`
	Name                   string `json:"name"`
	ParentModelId          string `json:"parentModelId"`
	ProblemId              string `json:"problemId"`
	SaveAnnotatedValImages bool   `json:"saveAnnotatedValImages"`
}

func (s *basicModelService) FineTune(ctx context.Context, req FineTuneRequestData) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)
		parentModel, build, problem := s.getParentModelBuildProblem(req.ParentModelId, req.BuildId, req.ProblemId)
		newModel, err := s.train(ctx, parentModel, build, problem, req.GpuNum, req.BatchSize, req.Epochs, req.Name)
		if err != nil {
			returnChan <- kitendpoint.Response{Data: nil, Err: err, IsLast: true}
			return
		}
		copySnapshotLatestToModelPath(newModel.TrainingWorkDir, newModel.SnapshotPath)
		copyTemplateYamlToNewModel(parentModel, newModel)
		copyConfigModelPy(newModel)
		newModel = s.eval(newModel, build, problem)
		newModel = s.fineTuneFinish(newModel)

		returnChan <- kitendpoint.Response{Data: newModel, Err: nil, IsLast: true}
	}()
	return returnChan
}

func copyConfigModelPy(model t.Model) {
	if err := copyFiles(fp.Join(model.TrainingWorkDir, "model.py"), model.ConfigPath); err != nil {
		log.Println("fine_tune.copyConfigModelPy.copyFiles(fp.Join(model.TrainingWorkDir, \"model.py\"), model.ConfigPath)", err)
	}
}

func copyTemplateYamlToNewModel(parentModel, newModel t.Model) {
	if err := copyFiles(parentModel.TemplatePath, newModel.TemplatePath); err != nil {
		log.Println("fine_tune.copyTemplateYamlToNewModel.copyFiles(parentModel.TemplatePath, newModel.TemplatePath)", err)
	}
}

func (s *basicModelService) train(ctx context.Context, parentModel t.Model, build t.Build, problem t.Problem, userGpuNum, batchSize, epochs int, newModelName string) (t.Model, error) {
	trainingWorkDir, _ := createTrainingWorkDir(problem.WorkingDir, newModelName)
	gpuNum := s.getOptimalGpuNumber(userGpuNum, parentModel.TrainingGpuNum)
	newModel, err := s.createNewModel(ctx, newModelName, problem, parentModel, trainingWorkDir, gpuNum, epochs)
	if err != nil {
		return newModel, err
	}
	commands := s.prepareFineTuneCommands(batchSize, gpuNum, newModel, parentModel, build, problem)
	outputLog := fmt.Sprintf("%s/output.log", newModel.Dir)
	env := getFineTuneEnv(parentModel)
	s.runCommand(commands, env, newModel.TrainingWorkDir, outputLog)
	return newModel, nil
}

func getFineTuneEnv(model t.Model) []string {
	return []string{
		fmt.Sprintf("MODEL_TEMPLATE_FILENAME=%s", model.TemplatePath),
		"MMDETECTION_DIR=/ote/external/mmdetection",
	}
}

func (s *basicModelService) fineTuneFinish(model t.Model) t.Model {
	model.Status = modelStatus.Finished
	modelUpdateOneRes := <-modelUpdateOne.Send(
		context.TODO(),
		s.Conn,
		model,
	)
	return modelUpdateOneRes.Data.(modelUpdateOne.ResponseData)
}

func (s *basicModelService) prepareFineTuneCommands(batchSize, gpuNum int, model, parentModel t.Model, build t.Build, problem t.Problem) []string {
	trainImgPrefixes, trainAnnFiles := s.getImgPrefixAndAnnotation("train", build, problem)
	valImgPrefixes, valAnnFiles := s.getImgPrefixAndAnnotation("val", build, problem)
	paramsArr := []string{
		fmt.Sprintf("--resume-from %s", parentModel.SnapshotPath),
		fmt.Sprintf("--train-ann-files %s", strings.Join(trainAnnFiles, ",")),
		fmt.Sprintf("--train-data-roots %s", strings.Join(trainImgPrefixes, ",")),
		fmt.Sprintf("--val-ann-files %s", strings.Join(valAnnFiles, ",")),
		fmt.Sprintf("--val-data-roots %s", strings.Join(valImgPrefixes, ",")),
		fmt.Sprintf("--save-checkpoints-to %s", model.TrainingWorkDir),
		fmt.Sprintf("--epochs %d", model.Epochs),
		fmt.Sprintf("--gpu-num %d", gpuNum),
	}

	if batchSize > 0 {
		paramsArr = append(paramsArr, fmt.Sprintf("--batch-size %d", batchSize))
	}

	paramsStr := strings.Join(paramsArr, " ")
	commands := []string{
		fmt.Sprintf(`pip install -e %s`, fp.Join(problem.ToolsPath, "packages", "ote")),
		fmt.Sprintf(`pip install -e %s`, fp.Join(problem.ToolsPath, "packages", "oteod")),
		fmt.Sprintf(`python %s %s`, parentModel.Scripts.Train, paramsStr),
	}
	return commands
}

func (s *basicModelService) getOptimalGpuNumber(gpuNum, parentGpuNum int) int {
	workerGpuNum := s.getTrainingWorkerGpuNum()
	log.Printf("getOptimalGpuNumber worker = %d, parent = %d, fromUI = %d", workerGpuNum, parentGpuNum, gpuNum)
	gpus := []int{gpuNum, parentGpuNum, workerGpuNum}
	gpus = arrays.FilterInt(gpus, func(a int) bool {
		return a > 0
	})
	result := arrays.MinInt(gpus)
	log.Printf("getOptimalGpuNumber result = %d", result)
	return result
}

func (s *basicModelService) getTrainingWorkerGpuNum() int {
	trainingWorkerGpuNumResp := <-trainingWorkerGpuNum.Send(
		context.TODO(),
		s.Conn,
		trainingWorkerGpuNum.RequestData{},
	)
	return trainingWorkerGpuNumResp.Data.(trainingWorkerGpuNum.ResponseData).Amount
}

func (s *basicModelService) runCommand(commands, env []string, workingDir, outputLog string) {
	<-runCommandsWorker.Send(
		context.Background(),
		s.Conn,
		runCommandsWorker.RequestData{
			Commands:  commands,
			OutputLog: outputLog,
			WorkDir:   workingDir,
			Env:       env,
		},
	)
}

func (s *basicModelService) createNewModel(
	ctx context.Context,
	name string,
	problem t.Problem,
	parentModel t.Model,
	trainingWorkDir string,
	gpuNum, epochs int,
) (t.Model, error) {
	dir := fp.Join(problem.Dir, name)
	if err := os.MkdirAll(dir, 0777); err != nil {
		log.Println("evaluate.createFolder.os.MkdirAll(path, 0777)", err)
	}
	tensorBoardLogDir := fp.Join(problem.WorkingDir, name)
	snapshotPath := fp.Join(dir, "snapshot.pth")
	configPath := fp.Join(dir, "model.py")
	newModelResp := <-modelInsertOne.Send(
		ctx,
		s.Conn,
		modelInsertOne.RequestData{
			ConfigPath:    configPath,
			Dir:           dir,
			Epochs:        parentModel.Epochs + epochs,
			Name:          name,
			ParentModelId: parentModel.Id,
			ProblemId:     problem.Id,
			SnapshotPath:  snapshotPath,
			Scripts: t.Scripts{
				Train: fp.Join(problem.ToolsPath, "train.py"),
				Eval:  fp.Join(problem.ToolsPath, "eval.py"),
			},
			Status:            modelStatus.InProgress,
			TemplatePath:      fp.Join(dir, "template.yaml"),
			TensorBoardLogDir: tensorBoardLogDir,
			TrainingGpuNum:    gpuNum,
			TrainingWorkDir:   trainingWorkDir,
		},
	)
	model := newModelResp.Data.(modelInsertOne.ResponseData)
	err := newModelResp.Err
	if err != nil {
		return model, newModelResp.Err.(error)
	} else {
		return model, nil
	}
}

func (s *basicModelService) getParentModelBuildProblem(modelIdString, buildIdString, problemIdString string) (t.Model, t.Build, t.Problem) {
	parentModelId, err := primitive.ObjectIDFromHex(modelIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	buildId, err := primitive.ObjectIDFromHex(buildIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	problemId, err := primitive.ObjectIDFromHex(problemIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	chParentModel := modelFindOne.Send(
		context.TODO(),
		s.Conn,
		modelFindOne.RequestData{Id: parentModelId},
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
	rParentModel, rBuild, rProblem := <-chParentModel, <-chBuild, <-chProblem
	parentModel := rParentModel.Data.(modelFindOne.ResponseData)
	build := rBuild.Data.(buildFindOne.ResponseData)
	problem := rProblem.Data.(problemFindOne.ResponseData)

	return parentModel, build, problem
}

func createTrainingWorkDir(problemWorkingDir, newModelName string) (string, error) {
	trainingWorkDir := fp.Join(problemWorkingDir, newModelName)
	log.Println("Create Dir:", trainingWorkDir)
	err := os.MkdirAll(trainingWorkDir, 0777)
	if err != nil {
		log.Println("MkdirAll", trainingWorkDir, err)
		return "", err
	}
	return trainingWorkDir, nil
}

func (s *basicModelService) createNewModelDir(problemFolder, newModelName string) string {
	newModelDirPath := fp.Join(problemFolder, newModelName)
	if err := os.Mkdir(newModelDirPath, 0777); err != nil {
		log.Println("fine_tune.createNewModelDir.os.Mkdir(newModelDirPath, 0777)", err)
	}
	return newModelDirPath
}

func (s *basicModelService) createNewBuildConfig(
	newModelDirPath, trainingWorkDir string,
	epochs int,
	parentModel t.Model,
	build t.Build,
	problem t.Problem,
) (string, error) {
	newModelConfigPath := fmt.Sprintf("%s/config.py", newModelDirPath)
	configPyFile, err := os.Open(parentModel.ConfigPath)
	if err != nil {
		fmt.Println("configPyFile", err)
	}
	defer configPyFile.Close()
	byteValue, _ := ioutil.ReadAll(configPyFile)

	defaultTotalEpochs := getVarValInt(byteValue, "total_epochs")
	byteValue = replaceVarValInt(byteValue, "total_epochs", defaultTotalEpochs+epochs)
	byteValue = replaceVarValStr(byteValue, "resume_from", parentModel.SnapshotPath)
	byteValue = replaceVarValStr(byteValue, "work_dir", trainingWorkDir)

	trainImgPrefixes, trainAnnFiles := s.getImgPrefixAndAnnotation("train", build, problem)
	byteValue = replaceVarValList(byteValue, "train_ann_file", stringArrToString(trainAnnFiles))
	byteValue = replaceVarValList(byteValue, "train_img_prefix", stringArrToString(trainImgPrefixes))

	valImgPrefixes, valAnnFiles := s.getImgPrefixAndAnnotation("val", build, problem)
	byteValue = replaceVarValList(byteValue, "val_ann_file", stringArrToString(valAnnFiles))
	byteValue = replaceVarValList(byteValue, "val_img_prefix", stringArrToString(valImgPrefixes))

	testImgPrefixes, testAnnFiles := s.getImgPrefixAndAnnotation("test", build, problem)
	byteValue = replaceVarValList(byteValue, "test_ann_file", stringArrToString(testAnnFiles))
	byteValue = replaceVarValList(byteValue, "test_img_prefix", stringArrToString(testImgPrefixes))

	err = ioutil.WriteFile(newModelConfigPath, byteValue, 0777)
	if err != nil {
		fmt.Println("WriteFile", err)
		return newModelConfigPath, err
	}
	return newModelConfigPath, nil
}

func (s *basicModelService) getImgPrefixAndAnnotation(category string, build t.Build, problem t.Problem) ([]string, []string) {
	trainAssetIds := getAssetsIdsList(category, build.Split["."].Children)

	cvatTaskFindResp := <-cvatTaskFind.Send(context.TODO(), s.Conn, cvatTaskFind.RequestData{ProblemId: problem.Id, AssetIds: trainAssetIds})
	cvatTasks := cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
	buildPath := fmt.Sprintf("%s/_builds/%s", problem.Dir, build.Folder)

	var annFiles []string
	var imgPrefixes []string
	for _, cvatTask := range cvatTasks {
		annFileName := fmt.Sprintf("%s.json", strconv.Itoa(cvatTask.Annotation.Id))
		annFilePath := fmt.Sprintf("%s/%s", buildPath, annFileName)
		annFiles = append(annFiles, annFilePath)
		asset := s.getAsset(cvatTask.AssetId)
		imgPrefixes = append(imgPrefixes, asset.CvatDataPath)
	}
	if len(annFiles) != len(imgPrefixes) {
		log.Println("domains.model.pkg.service.fine_tune.getImgPrefixAndAnnotation: len(annFiles) != len(imgPrefixes)")
	}
	return imgPrefixes, annFiles
}

func (s *basicModelService) getAsset(assetId primitive.ObjectID) t.Asset {
	assetFindOneResp := <-assetFindOne.Send(context.TODO(), s.Conn, assetFindOne.RequestData{Id: assetId})
	return assetFindOneResp.Data.(assetFindOne.ResponseData)
}

func stringArrToString(arr []string) string {
	b, err := json.Marshal(arr)
	if err != nil {
		log.Println("domains.model.pkg.service.fine_tune.stringArrToString.json.Marshal(arr)", err)
		return ""
	}
	return string(b)
}

func getAssetsIdsList(category string, buildSplit map[string]t.BuildAssetsSplit) []primitive.ObjectID {
	var result []primitive.ObjectID
	for _, child := range buildSplit {
		if len(child.Children) == 0 && isCategory(child, category) {
			result = append(result, child.AssetId)
		} else {
			childAssetIdsList := getAssetsIdsList(category, child.Children)
			result = append(result, childAssetIdsList...)
		}
	}
	return result
}

func isCategory(split t.BuildAssetsSplit, category string) bool {
	category = strings.ToLower(category)
	if split.Train == splitState.Confirmed && category == "train" {
		return true
	} else if split.Test == splitState.Confirmed && category == "test" {
		return true
	} else if split.Val == splitState.Confirmed && category == "val" {
		return true
	}
	return false

}

func replaceVarValList(src []byte, name string, value string) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*['"]?.*['"]?[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = %s`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func replaceVarValStr(src []byte, name string, value string) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*['"]?.*['"]?[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = "%s"`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func replaceVarValInt(src []byte, name string, value int) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*.*[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = %d`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func getVarValInt(src []byte, name string) int {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*(.*)[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	submatch := r.FindSubmatch(src)
	intRes, err := strconv.Atoi(string(submatch[1]))
	if err != nil {
		log.Println("Atoi", err)
	}
	return intRes
}

func copySnapshotLatestToModelPath(trainingPath, newSnapshotPath string) {
	snapshotPath := fmt.Sprintf("%s/latest.pth", trainingPath)
	if _, err := ufiles.Copy(snapshotPath, newSnapshotPath); err != nil {
		log.Println("copySnapshotLatestToModelPath.Copy", err)
	}
}
