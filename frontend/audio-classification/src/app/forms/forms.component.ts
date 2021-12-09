import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';

@Component({
  selector: 'app-forms',
  templateUrl: './forms.component.html',
  styleUrls: ['./forms.component.scss'],
})
export class FormsComponent implements OnInit {
  form!: FormGroup;

  constructor(
    private httpClient: HttpClient,
    private formBuilder: FormBuilder
  ) {}

  ngOnInit(): void {
    this.initForm();
  }

  initForm(): void {
    this.form = this.formBuilder.group({});
  }
  onSubmit(): void {
    console.log(this.form.value);
    /*
    this.httpClient.post("http://localhost:5000/svm",this.form.value).subscribe(
      (response: any) => {
        console.log(response.genre);

      },
      (error)=>{

      }
    )*/
  }
}
