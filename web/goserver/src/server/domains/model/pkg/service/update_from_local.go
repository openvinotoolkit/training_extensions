package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	fp "path/filepath"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"gopkg.in/yaml.v2"

	buildFindOne "server/db/pkg/handler/build/find_one"
	buildInsertOne "server/db/pkg/handler/build/insert_one"
	modelUpdateUpsert "server/db/pkg/handler/model/update_upsert"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
	modelStatus "server/db/pkg/types/status/model"
	kitendpoint "server/kit/endpoint"
	u "server/kit/utils"
	uFiles "server/kit/utils/basic/files"
)

type Basic struct {
	BatchSize        int     `yaml:"batch_size"`
	BaseLearningrate float64 `yaml:"base_learning_rate"`
	Epochs           int     `yaml:"epochs"`
}

type HyperParameters struct {
	Basic Basic `yaml:"basic"`
}

type ModelYml struct {
	Class           string          `yaml:"domain"`
	Name            string          `yaml:"name"`
	Problem         string          `yaml:"problem"`
	Dependencies    []t.Dependency  `yaml:"dependencies"`
	Metrics         []t.Metric      `yaml:"metrics"`
	GpuNum          int             `yaml:"gpu_num"`
	Config          string          `yaml:"config"`
	HyperParameters HyperParameters `yaml:"hyper_parameters"`
}

type UpdateFromLocalRequestData struct {
	Path string `json:"path"`
}

func (s *basicModelService) UpdateFromLocal(ctx context.Context, req UpdateFromLocalRequestData) chan kitendpoint.Response {
	responseChan := make(chan kitendpoint.Response)
	go func() {
		modelYml := getModelYaml(req.Path)
		problem := s.getProblem(ctx, modelYml.Problem)
		defaultBuild := s.getDefaultBuild(problem.Id)
		model := s.prepareModel(modelYml, defaultBuild.Id, problem)
		copyModelFiles(fp.Dir(req.Path), model.Dir, req.Path, problem.ToolsPath, modelYml)
		model = s.updateCreateModel(model)
		responseChan <- kitendpoint.Response{Data: model, Err: nil, IsLast: true}
	}()
	return responseChan
}

func copyModelFiles(from, to, modelTemplatePath, toolsPath string, modelYml ModelYml) {
	copyConfig(from, to, modelYml)
	copyDependencies(from, to, toolsPath, modelYml)
	saveMetrics(to, modelYml)
	copyTemplateYaml(modelTemplatePath, to)
}

func copyConfig(from, to string, modelYml ModelYml) {
	if err := copyFiles(fp.Join(from, modelYml.Config), fp.Join(to, modelYml.Config)); err != nil {
		log.Println("update_from_local.copyDependencies.copyFiles(fp.Join(from, modelYml.Config), fp.Join(to, modelYml.Config))", err)
	}
}

func copyTemplateYaml(from, to string) {
	if err := copyFiles(from, fp.Join(to, "template.yaml")); err != nil {
		log.Println("update_from_local.copyDependencies.copyFiles(fp.Join(from, modelYml.Config), fp.Join(to, modelYml.Config))", err)
	}
}

func copyDependencies(from, to, toolsPath string, modelYml ModelYml) {
	for _, d := range modelYml.Dependencies {
		toDir := dirByDestination(to, toolsPath, d.Destination)
		if isValidUrl(d.Source) {
			if err := downloadWithCheck(d.Source, toDir, d.Sha256, d.Size); err != nil {
				log.Println("update_from_local.copyDependencies.downloadWithCheck(d.Source, d.Destination, d.Sha256, d.Size)", err)
			}
		} else {
			if err := copyFiles(fp.Join(from, d.Source), toDir); err != nil {
				log.Println("update_from_local.copyDependencies.copyFiles(fp.Join(from, d.Source), fp.Join(to, d.Destination))", err)
			}
		}
	}
}

func saveMetrics(to string, modelYml ModelYml) {
	metricsPath := fp.Join(to, "_default", "metrics.yaml")
	if err := os.MkdirAll(fp.Dir(metricsPath), 0777); err != nil {
		log.Println("saveMetrics.os.MkdirAll(fp.Dir(metricsPath), 0777)", err)
	}
	f, err := os.Create(metricsPath)
	if err != nil {
		log.Println("saveMetrics.os.Create(metricsPath)", err)
	}
	metrics, err := yaml.Marshal(modelYml.Metrics)
	if err != nil {
		log.Println("saveMetrics.yaml.Marshal(modelYml.Metrics)", err)
	}
	_, err = f.Write(metrics)
	if err != nil {
		log.Println("saveMetrics.f.Write(metrics)", err)
	}
	if err := f.Sync(); err != nil {
		log.Println("saveMetrics.f.Sync()", err)
	}
}

func dirByDestination(to, toolsPath, dest string) string {
	if dest == "snapshot.pth" {
		return fp.Join(to, dest)
	}
	return fp.Join(toolsPath, dest)
}

func copyFiles(from, to string) error {
	si, err := os.Stat(from)
	if err != nil {
		log.Println("update_from_local.copyFiles.os.Stat(from)", err)
		return err
	}
	if si.IsDir() {
		if err := uFiles.CopyDir(from, to); err != nil {
			log.Println("update_from_local.copyFiles.uFiles.CopyDir(from, to)", err)
			return err
		}
	} else {
		if _, err := uFiles.Copy(from, to); err != nil {
			log.Println("update_from_local.copyFiles.uFiles.Copy(from, to)", err)
			return err
		}
	}
	return nil
}

func isValidUrl(toTest string) bool {
	_, err := url.ParseRequestURI(toTest)
	if err != nil {
		return false
	}

	u, err := url.Parse(toTest)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return false
	}

	return true
}

func (s *basicModelService) prepareModel(modelYml ModelYml, buildId primitive.ObjectID, problem t.Problem) t.Model {
	modelFolderName := u.StringToFolderName(modelYml.Name)
	dir := fp.Join(problem.Dir, modelFolderName)
	metrics := make(map[string][]t.Metric)
	metrics[buildId.Hex()] = modelYml.Metrics
	model := t.Model{
		BatchSize:   modelYml.HyperParameters.Basic.BatchSize,
		ConfigPath:  fp.Join(dir, modelYml.Config),
		ProblemId:   problem.Id,
		Description: "",
		Dir:         dir,
		Epochs:      modelYml.HyperParameters.Basic.Epochs,
		Metrics:     metrics,
		Name:        modelYml.Name,
		Scripts: t.Scripts{
			Train: fp.Join(problem.ToolsPath, "train.py"),
			Eval:  fp.Join(problem.ToolsPath, "eval.py"),
		},
		SnapshotPath:      fp.Join(dir, "snapshot.pth"),
		Status:            modelStatus.Default,
		TemplatePath:      fp.Join(dir, "template.yaml"),
		TensorBoardLogDir: "",
		TrainingGpuNum:    modelYml.GpuNum,
		TrainingWorkDir:   "",
	}
	log.Println("Epochs:", modelYml.HyperParameters.Basic.Epochs, model.Epochs)
	return model
}

func getModelYaml(path string) (modelYml ModelYml) {
	yamlFile, err := ioutil.ReadFile(path)
	if err != nil {
		log.Println("ReadFile", err)
	}
	err = yaml.Unmarshal(yamlFile, &modelYml)

	if err != nil {
		log.Println("Unmarshal", err)
	}
	return modelYml
}

func (s *basicModelService) getProblem(ctx context.Context, title string) t.Problem {
	problemResp := <-problemFindOne.Send(
		ctx,
		s.Conn,
		problemFindOne.RequestData{
			Title: title,
		},
	)
	return problemResp.Data.(problemFindOne.ResponseData)
}

func downloadWithCheck(url, dst, sha256 string, size int) error {
	for true {
		nBytes, err := u.DownloadFile(url, dst)
		if err != nil {
			log.Println("downloadWithCheck.DownloadFile", err)
			continue
		}
		log.Println(dst, nBytes)
		// TODO: uncommit
		// if nBytes != int64(size) {
		// 	log.Println("downloadWithCheck.WrongSize", err)
		// 	err = errors.New("wrong size")
		// 	continue
		// }
		// dstSha265 := getSha265(dst)
		// if dstSha265 != sha256 {
		// 	log.Println("downloadWithCheck.WrongSha", err)
		// 	err = errors.New("wrong sha")
		// 	continue
		// }
		break
	}
	return nil

}

func getSha265(path string) string {
	f, err := os.Open(path)
	if err != nil {
		log.Println("getSha265.os.Open(path)", err)
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Println("getSha265.io.Copy(h, f)", err)
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (s *basicModelService) updateCreateModel(model t.Model) t.Model {
	log.Println("updateCreateModel.Epochs", model.Epochs)
	modelResp := <-modelUpdateUpsert.Send(
		context.TODO(),
		s.Conn,
		modelUpdateUpsert.RequestData{
			ConfigPath:     model.ConfigPath,
			ProblemId:      model.ProblemId,
			Description:    model.Description,
			Dir:            model.Dir,
			Epochs:         model.Epochs,
			Framework:      model.Framework,
			Metrics:        model.Metrics,
			Name:           model.Name,
			Scripts:        model.Scripts,
			SnapshotPath:   model.SnapshotPath,
			Status:         model.Status,
			TemplatePath:   model.TemplatePath,
			TrainingGpuNum: model.TrainingGpuNum,
		},
	)
	return modelResp.Data.(modelUpdateUpsert.ResponseData)
}

func (s *basicModelService) getDefaultBuild(problemId primitive.ObjectID) (result t.Build) {
	buildFindOneResp := <-buildFindOne.Send(
		context.TODO(),
		s.Conn,
		buildFindOne.RequestData{
			ProblemId: problemId,
			Name:      "default",
		},
	)
	result = buildFindOneResp.Data.(buildFindOne.ResponseData)
	if result.Id.IsZero() {
		buildInsertOneResp := <-buildInsertOne.Send(
			context.TODO(),
			s.Conn,
			buildInsertOne.RequestData{
				ProblemId: problemId,
				Name:      "default",
				Status:    buildStatus.Default,
			},
		)
		result = buildInsertOneResp.Data.(buildInsertOne.ResponseData)
	}
	return result
}
