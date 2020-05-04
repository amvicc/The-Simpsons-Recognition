import React from 'react';
import axios from 'axios'
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import CircularProgress from '@material-ui/core/CircularProgress';

class Main extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      imageURL: '',
      image: null,
      name: ' ',
      isLoading: false
    };

    this.handleUploadImage = this.handleUploadImage.bind(this);
    this.handleChange = this.handleChange.bind(this)
  }

  handleUploadImage(event) {
    event.preventDefault();

    const data = new FormData();
    data.append('file', this.state.image)
    
    this.setState({
        ...this.state.image,
        ...this.state.imageURL,
        ...this.state.name,
        isLoading: true
    })

    axios({
        method: 'POST',
        url: '/upload',
        data: data
    })
    .then( (response) => {
        let name = response.data
        name = name.replace('_', ' ')
        const toTitleCase = (phrase) => {
            return phrase
              .toLowerCase()
              .split(' ')
              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
              .join(' ');
        };
        let result = "This is " + toTitleCase(name);
        
        this.setState({
            ...this.state.image,
            ...this.state.imageURL,
            name: result,
            isLoading: false
        })
    })
    .catch((err) => {
        this.setState({
            ...this.state.image,
            ...this.state.imageURL,
            ...this.state.name,
            isLoading: false
        })
        alert("Server error:" + err)
    })
  }

  handleChange(event) {
      if(event.target !== undefined) {
        this.setState({
            imageURL: URL.createObjectURL(event.target.files[0]),
            image: event.target.files[0],
            ...this.state.name
        })
      }
  }

  render() {
    return (
        <Box width="100%" style={{margin: 75}}>
            {this.state.isLoading?(
                <CircularProgress />
            ):(
                <Grid 
                    container 
                    spacing={3} 
                    direction="row"
                    justify="center"
                    alignItems="center"
                >
                    <Grid item xs={3}>
                        <input 
                            onChange={this.handleChange} 
                            type="file"
                            id="contained-button-file"
                            style={{ display: "none" }}
                        />
                        <label htmlFor="contained-button-file">
                            <Button style={{marginTop: '25px', width: '315px'}} variant="contained" color="primary" component="span"> 
                                Загрузить фотографию Симпсона
                            </Button>
                        </label>
                        <Button style={{marginTop: '25px', width: '315px'}} variant="contained" color="primary" onClick={this.handleUploadImage}>
                            Узнать имя Симпсона
                        </Button>
                        {this.state.name === ' '?(
                            <div></div>
                        ):(
                            <TextField
                                id="standard-read-only-input"
                                value={this.state.name}
                                InputProps={{
                                    readOnly: true,
                                }}
                                variant="outlined"
                                style={{marginTop: '25px', width: '315px', textAlign: 'center'}}
                            />
                        )}
                    </Grid>
                    <Grid item xs={3}>
                        {this.state.imageURL === ''?(
                            <div></div>
                        ):(
                            <img style={{width: '400px'}} src={this.state.imageURL} alt="img"/>
                        )}
                    </Grid>
                </Grid>
            )}
        </Box>
    );
  }
}

export default Main;