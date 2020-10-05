package service

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"

	"gopkg.in/yaml.v2"

	problemUpdateUpsert "server/db/pkg/handler/problem/update_upsert"
	t "server/db/pkg/types"
	problemType "server/db/pkg/types/problem/types"
	kitendpoint "server/kit/endpoint"
	u "server/kit/utils"
)

type UpdateFromLocalRequestData struct {
	Path string `json:"path"`
}

func (s *basicProblemService) UpdateFromLocal(
	ctx context.Context,
	req UpdateFromLocalRequestData,
	responseChan chan kitendpoint.Response,
) {
	var problems []t.Problem
	domains := readProblemsYaml(responseChan, req.Path)
	for _, domain := range domains {
		for _, problemData := range domain.Problems {
			problem := s.createOrUpdateProblem(ctx, problemData, domain.Title)
			if err := os.MkdirAll(problem.Dir, 0777); err != nil {
				log.Println("UpdateFromLocal.os.MkdirAll(problem.Dir, 0777)", err)
			}
			problems = append(problems, problem)
			log.Println(problems)
		}
	}
	log.Println(problems)
	responseChan <- kitendpoint.Response{Data: problems, IsLast: true, Err: nil}
}

func readProblemsYaml(responseChan chan kitendpoint.Response, path string) []t.Domain {
	type Domain struct {
		Problems []t.Problem `yaml:"problems"`
		Title    string      `yaml:"title"`
	}
	result := struct {
		Domains []t.Domain `yaml:"domains"`
	}{}
	yamlFile, err := ioutil.ReadFile(path)
	if err != nil {
		log.Println("ReadFile", err)
		responseChan <- kitendpoint.Response{
			Data:   t.Problem{},
			IsLast: true,
			Err:    err,
		}
	}
	err = yaml.Unmarshal(yamlFile, &result)
	if err != nil {
		log.Println("Unmarshal", err)
		responseChan <- kitendpoint.Response{
			Data:   t.Problem{},
			IsLast: true,
			Err:    err,
		}
	}
	for di, domain := range result.Domains {
		for pi, problem := range domain.Problems {
			if err := json.Unmarshal([]byte(problem.CvatSchema), &result.Domains[di].Problems[pi].Labels); err != nil {
				log.Println("json.Unmarshal([]byte(problem.CvatSchema), &problem.Labels)", err)
			}
		}
	}

	return result.Domains
}

func (s *basicProblemService) createOrUpdateProblem(ctx context.Context, problemData t.Problem, domainTitle string) t.Problem {
	problemFolderName := u.StringToFolderName(problemData.Title)
	classFolderName := u.StringToFolderName(domainTitle)
	problemDir := fp.Join(s.problemPath, classFolderName, problemFolderName)
	workingDir := fp.Join(s.trainingsPath, classFolderName, problemFolderName)
	if problemData.Type == "" {
		problemData.Type = problemType.Default
	}
	requestData := problemUpdateUpsert.RequestData{
		Class:       domainTitle,
		Description: problemData.Description,
		ImagesUrls:  problemData.ImagesUrls,
		Dir:         problemDir,
		Labels:      problemData.Labels,
		Subtitle:    problemData.Subtitle,
		Title:       problemData.Title,
		Type:        problemData.Type,
		WorkingDir:  workingDir,
	}

	problemUpdateUpsertResp := <-problemUpdateUpsert.Send(
		ctx,
		s.Conn,
		requestData,
	)
	return problemUpdateUpsertResp.Data.(problemUpdateUpsert.ResponseData)
}
