package service

import (
	"context"
	"log"
	"os"
	"os/exec"

	"github.com/mattn/go-shellwords"
)

type RunCommandsRequestData struct {
	WorkDir   string   `json:"workDir"`
	Commands  []string `json:"commands"`
	OutputLog string   `json:"outputLog"`
	Env       []string `json:"env"`
}

func (s *basicTrainModelService) RunCommands(ctx context.Context, req RunCommandsRequestData) (interface{}, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	f, err := os.Create(req.OutputLog)
	if err != nil {
		log.Println(err)
	}
	for _, command := range req.Commands {
		log.Println(command, "started")
		cmdArr, err := shellwords.Parse(command)
		if err != nil {
			log.Println("shellwords.Parse(command)", err)
		}
		cmdName := cmdArr[0]
		cmdArgs := cmdArr[1:]
		cmd := exec.Command(cmdName, cmdArgs...)
		cmd.Env = append(os.Environ(), req.Env...)
		cmd.Dir = req.WorkDir
		cmd.Stdout = f
		cmd.Stderr = f
		err = cmd.Run()
		if err != nil {
			log.Printf("cmd.Run() failed with %s\n", err)
		}
		log.Println(command, "finished")
	}
	return nil, nil
}
