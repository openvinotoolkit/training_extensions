package service

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"os"
	fp "path/filepath"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	"github.com/google/uuid"

	modelFind "server/db/pkg/handler/model/find"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	problemUpdateUpsert "server/db/pkg/handler/problem/update_upsert"
	t "server/db/pkg/types"
	problemType "server/db/pkg/types/problem/types"
	modelCreateFromGeneric "server/domains/model/pkg/handler/create_from_generic"
	kitendpoint "server/kit/endpoint"
	u "server/kit/utils"
)

type CreateRequestData struct {
	Class       string                   `bson:"class" json:"class"`
	Description string                   `bson:"description" json:"description"`
	Image       string                   `bson:"image" json:"image"`
	Labels      []map[string]interface{} `bson:"labels" json:"labels"`
	Subtitle    string                   `bson:"subtitle" json:"subtitle"`
	Title       string                   `bson:"title" json:"title"`
}

func (s *basicProblemService) Create(ctx context.Context, req CreateRequestData, responseChan chan kitendpoint.Response) {
	genericProblem := s.getGenericProblem(req.Class)
	imageUrl := saveImage(req.Image)
	problemDir := s.createProblemDir(req.Class, req.Title)
	problemCreateRequest := getNewProblemCreateRequest(genericProblem, imageUrl, req.Title, req.Subtitle, req.Description, problemDir, req.Class, req.Labels)
	problem := s.createProblem(problemCreateRequest)
	responseChan <- kitendpoint.Response{Data: problem, IsLast: true, Err: nil}

	for _, genericModel := range s.getGenericModels(genericProblem.Id) {
		modelCreateFromGeneric.Send(context.TODO(), s.Conn, modelCreateFromGeneric.RequestData{
			GenericModelId: genericModel.Id,
			ProblemId:      problem.Id,
		})
	}
}

func (s *basicProblemService) getGenericProblem(class string) t.Problem {
	problemFindOneResp := <-problemFindOne.Send(context.TODO(), s.Conn, problemFindOne.RequestData{Class: class, Type: problemType.Generic})
	return problemFindOneResp.Data.(problemFindOne.ResponseData)
}

func (s *basicProblemService) getGenericModels(problemId primitive.ObjectID) []t.Model {
	modelFindResp := <-modelFind.Send(context.TODO(), s.Conn, modelFind.RequestData{
		ProblemId: problemId,
	})
	return modelFindResp.Data.(modelFind.ResponseData).Items
}

func (s *basicProblemService) createProblemDir(class, title string) string {
	problemFolderName := u.StringToFolderName(title)
	classFolderName := u.StringToFolderName(class)
	problemDir := fp.Join(s.problemPath, classFolderName, problemFolderName)
	if err := os.MkdirAll(problemDir, 0777); err != nil {
		log.Println(err)
	}
	return problemDir
}

func getNewProblemCreateRequest(genericProblem t.Problem, imageUrl, title, subtitle, description, folder, class string, labels []map[string]interface{}) problemUpdateUpsert.RequestData {
	var imagesUrls []string
	if imageUrl != "" {
		imagesUrls = append(imagesUrls, imageUrl)
	}
	return problemUpdateUpsert.RequestData{
		Class:       class,
		Description: description,
		ImagesUrls:  imagesUrls,
		Dir:         folder,
		Labels:      labels,
		Subtitle:    subtitle,
		Title:       title,
		Type:        problemType.Custom,
		WorkingDir:  genericProblem.WorkingDir,
	}
}

func saveImage(imageString string) string {
	if imageString == "" {
		return ""
	}
	split := strings.Split(imageString, ",")
	ext := strings.Split(strings.Split(strings.Split(split[0], ":")[1], ";")[0], "/")[1]
	imageB64 := split[1]
	dec, err := base64.StdEncoding.DecodeString(imageB64)
	if err != nil {
		panic(err)
	}
	imageName := uuid.New().String() + "." + ext
	imagePath := "/media/" + imageName
	f, err := os.Create(imagePath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	if _, err := f.Write(dec); err != nil {
		panic(err)
	}
	if err := f.Sync(); err != nil {
		panic(err)
	}
	return "/images/" + imageName
}

func (s *basicProblemService) createProblem(req problemUpdateUpsert.RequestData) t.Problem {
	problemUpdateUpsertResp := <-problemUpdateUpsert.Send(
		context.TODO(),
		s.Conn,
		req,
	)
	return problemUpdateUpsertResp.Data.(problemUpdateUpsert.ResponseData)
}

func UnmarshalLabels(labelsString string) (res []map[string]interface{}) {
	err := json.Unmarshal([]byte(labelsString), &res)
	if err != nil {
		log.Println("domains.problem.pkg.service.create.json.Unmarshal([]byte(labelsString), &res)", err)
	}
	return res
}
